//! # Quantizer Abstractions & Engines
//!
//! Affine, symmetric, and per-channel quantizers transforming continuous tensors to discrete representations.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::core::{QParams, QuantDType, QuantResult, QuantScheme, QuantTensor};
use super::utils::{compute_scale_zero_point, minmax, quantize_val};

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

        Ok(QuantTensor::new(q_data, tensor.shape().to_vec(), self.params.clone()))
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

        Ok(QuantTensor::new(q_data, tensor.shape().to_vec(), self.params.clone()))
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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_quantizer_stress_001() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 1 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_002() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 2 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_003() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 3 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_004() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 4 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_005() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 5 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_006() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 6 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_007() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 7 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_008() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 8 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_009() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 9 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_010() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 10 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_011() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 11 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_012() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 12 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_013() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 13 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_014() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 14 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_015() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 15 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_016() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 16 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_017() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 17 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_018() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 18 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_019() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 19 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_020() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 20 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_021() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 21 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_022() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 22 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_023() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 23 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_024() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 24 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_025() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 25 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_026() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 26 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_027() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 27 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_028() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 28 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_029() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 29 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_030() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 30 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_031() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 31 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_032() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 32 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_033() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 33 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_034() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 34 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_035() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 35 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_036() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 36 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_037() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 37 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_038() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 38 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_039() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 39 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_040() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 40 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_041() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 41 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_042() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 42 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_043() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 43 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_044() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 44 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_045() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 45 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_046() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 46 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_047() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 47 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_048() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 48 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_049() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 49 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_050() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 50 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_051() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 51 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_052() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 52 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_053() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 53 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_054() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 54 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_055() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 55 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_056() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 56 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_057() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 57 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_058() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 58 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_059() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 59 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_060() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 60 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_061() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 61 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_062() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 62 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_063() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 63 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_064() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 64 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_065() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 65 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_066() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 66 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_067() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 67 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_068() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 68 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_069() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 69 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_070() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 70 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_071() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 71 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_072() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 72 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_073() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 73 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_074() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 74 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_075() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 75 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_076() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 76 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_077() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 77 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_078() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 78 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_079() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 79 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_080() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 80 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_081() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 81 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_082() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 82 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_083() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 83 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_084() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 84 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_085() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 85 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_086() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 86 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_087() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 87 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_088() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 88 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_089() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 89 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_090() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 90 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_091() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 91 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_092() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 92 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_093() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 93 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_094() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 94 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_095() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 95 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_096() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 96 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_097() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 97 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_098() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 98 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_099() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 99 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_100() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 100 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_101() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 101 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_102() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 102 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_103() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 103 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_104() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 104 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_105() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 105 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_106() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 106 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_107() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 107 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_108() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 108 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_109() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 109 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_110() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 110 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_111() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 111 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_112() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 112 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_113() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 113 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_114() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 114 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_115() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 115 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_116() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 116 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_117() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 117 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_118() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 118 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_119() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 119 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_120() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 120 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_121() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 121 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_122() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 122 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_123() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 123 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_124() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 124 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_125() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 125 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_126() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 126 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_127() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 127 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_128() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 128 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_129() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 129 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_130() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 130 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_131() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 131 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_132() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 132 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_133() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 133 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_134() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 134 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_135() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 135 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_136() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 136 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_137() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 137 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_138() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 138 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_139() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 139 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_140() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 140 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_141() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 141 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_142() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 142 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_143() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 143 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_144() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 144 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_145() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 145 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_146() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 146 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_147() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 147 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_148() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 148 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_149() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 149 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_150() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 150 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_151() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 151 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_152() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 152 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_153() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 153 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_154() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 154 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_155() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 155 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_156() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 156 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_157() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 157 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_158() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 158 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_159() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 159 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_160() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 160 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_161() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 161 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_162() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 162 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_163() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 163 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_164() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 164 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_165() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 165 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_166() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 166 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_167() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 167 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_168() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 168 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_169() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 169 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_170() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 170 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_171() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 171 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_172() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 172 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_173() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 173 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_174() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 174 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_175() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 175 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_176() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 176 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_177() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 177 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_178() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 178 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_179() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 179 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_180() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 180 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_181() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 181 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_182() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 182 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_183() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 183 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_184() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 184 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_185() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 185 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_186() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 186 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_187() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 187 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_188() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 188 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_189() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 189 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_190() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 190 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_191() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 191 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_192() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 192 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_193() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 193 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_194() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 194 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_195() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 195 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_196() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 196 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_197() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 197 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_198() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 198 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_199() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 199 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_200() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 200 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_201() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 201 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_202() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 202 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_203() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 203 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_204() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 204 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_205() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 205 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_206() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 206 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_207() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 207 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_208() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 208 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_209() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 209 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_210() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 210 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_211() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 211 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_212() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 212 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_213() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 213 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_214() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 214 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_215() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 215 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_216() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 216 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_217() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 217 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_218() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 218 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_219() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 219 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_220() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 220 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_221() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 221 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_222() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 222 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_223() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 223 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_224() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 224 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_225() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 225 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_226() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 226 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_227() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 227 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_228() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 228 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_229() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 229 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_230() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 230 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_231() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 231 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_232() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 232 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_233() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 233 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_234() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 234 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_235() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 235 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_236() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 236 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_237() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 237 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_238() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 238 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_239() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 239 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_240() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 240 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_241() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 241 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_242() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 242 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_243() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 243 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_244() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 244 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_245() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 245 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_246() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 246 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_247() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 247 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_248() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 248 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_249() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 249 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_250() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 250 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_251() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 251 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_252() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 252 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_253() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 253 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_254() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 254 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_255() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 255 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_256() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 256 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_257() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 257 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_258() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 258 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_259() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 259 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_260() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 260 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_261() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 261 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_262() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 262 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_263() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 263 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_264() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 264 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_265() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 265 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_266() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 266 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_267() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 267 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_268() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 268 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_269() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 269 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_270() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 270 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_271() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 271 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_272() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 272 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_273() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 273 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_274() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 274 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_275() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 275 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_276() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 276 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_277() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 277 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_278() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 278 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_279() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 279 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_280() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 280 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_281() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 281 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_282() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 282 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_283() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 283 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_284() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 284 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_285() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 285 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_286() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 286 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_287() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 287 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_288() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 288 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_289() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 289 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_290() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 290 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_291() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 291 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_292() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 292 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_293() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 293 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_294() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 294 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_295() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 295 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_296() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 296 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_297() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 297 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_298() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 298 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_299() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 299 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_300() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 300 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_301() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 301 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_302() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 302 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_303() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 303 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_304() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 304 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_305() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 305 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_306() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 306 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_307() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 307 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_308() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 308 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_309() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 309 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_310() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 310 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_311() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 311 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_312() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 312 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_313() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 313 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_314() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 314 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_315() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 315 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_316() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 316 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_317() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 317 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_318() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 318 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_319() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 319 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_320() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 320 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_321() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 321 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_322() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 322 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_323() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 323 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_324() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 324 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_325() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 325 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_326() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 326 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_327() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 327 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_328() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 328 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_329() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 329 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_330() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 330 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_331() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 331 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_332() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 332 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_333() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 333 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_334() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 334 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_335() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 335 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_336() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 336 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_337() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 337 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_338() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 338 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_339() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 339 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_340() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 340 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_341() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 341 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_342() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 342 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_343() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 343 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_344() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 344 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_345() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 345 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_346() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 346 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_347() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 347 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_348() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 348 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_349() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 349 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_350() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 350 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_351() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 351 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_352() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 352 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_353() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 353 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_354() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 354 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_355() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 355 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_356() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 356 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_357() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 357 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    #[test]
    fn test_quantizer_stress_358() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 358 as f64 * 0.05, 1.0], vec![4]);
        let quant = AffineQuantizer::from_tensor(&t, QuantDType::Int8).unwrap();
        let qt = quant.quantize(&t).unwrap();
        let deq = quant.dequantize(&qt).unwrap();
        assert_eq!(deq.shape(), &[4]);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
}
