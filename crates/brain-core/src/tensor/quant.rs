//! 8-bit integer quantization (INT8/UINT8), affine scale/zero-point maps, and quantized inference kernels.
//!
//! This module provides [`QuantizedTensor`], per-tensor/per-channel quantization/dequantization, and quantized linear operations.

use crate::tensor::Tensor;

/// An 8-bit affine quantized tensor.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    data: Vec<i8>,
    shape: Vec<usize>,
    scale: f64,
    zero_point: i32,
}

impl QuantizedTensor {
    /// Creates a new quantized tensor.
    pub fn new(data: Vec<i8>, shape: Vec<usize>, scale: f64, zero_point: i32) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        QuantizedTensor {
            data,
            shape,
            scale,
            zero_point,
        }
    }

    /// Returns data slice.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Returns shape slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns scale.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns zero point.
    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }
}

/// Quantizes a float tensor to 8-bit signed integer format: q = clamp(round(x / scale) + zero_point, -128, 127).
pub fn quantize_per_tensor(input: &Tensor, scale: f64, zero_point: i32) -> QuantizedTensor {
    assert!(scale > 0.0, "quantize scale must be > 0");
    let inv_scale = 1.0 / scale;
    let data: Vec<i8> = input
        .data()
        .iter()
        .map(|&x| {
            let q = (x * inv_scale).round() as i32 + zero_point;
            q.max(-128).min(127) as i8
        })
        .collect();
    QuantizedTensor::new(data, input.shape().to_vec(), scale, zero_point)
}

/// Dequantizes an 8-bit tensor back to floating point: x = (q - zero_point) * scale.
pub fn dequantize_per_tensor(q: &QuantizedTensor) -> Tensor {
    let data: Vec<f64> = q
        .data
        .iter()
        .map(|&val| ((val as i32) - q.zero_point) as f64 * q.scale)
        .collect();
    Tensor::new(data, q.shape.clone())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], vec![4]);
        let q = quantize_per_tensor(&t, 0.05, 0);
        let restored = dequantize_per_tensor(&q);
        for (a, b) in t.data().iter().zip(restored.data().iter()) {
            assert!((a - b).abs() <= 0.05);
        }
    }

    #[test]
    fn test_quant_symmetric_affine() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0], vec![3]);
        let q = quantize_per_tensor(&x, 0.01, 0);
        let dq = dequantize_per_tensor(&q);
        for i in 0..3 {
            assert!((dq.get(i) - x.get(i)).abs() < 0.05);
        }
    }
}
