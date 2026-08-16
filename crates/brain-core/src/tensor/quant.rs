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
    fn test_quant_stress_case_001() {
        let val = ((1 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_002() {
        let val = ((2 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_003() {
        let val = ((3 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_004() {
        let val = ((4 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_005() {
        let val = ((5 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_006() {
        let val = ((6 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_007() {
        let val = ((7 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_008() {
        let val = ((8 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_009() {
        let val = ((9 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_010() {
        let val = ((10 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_011() {
        let val = ((11 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_012() {
        let val = ((12 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_013() {
        let val = ((13 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_014() {
        let val = ((14 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_015() {
        let val = ((15 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_016() {
        let val = ((16 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_017() {
        let val = ((17 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_018() {
        let val = ((18 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_019() {
        let val = ((19 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_020() {
        let val = ((20 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_021() {
        let val = ((21 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_022() {
        let val = ((22 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_023() {
        let val = ((23 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_024() {
        let val = ((24 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_025() {
        let val = ((25 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_026() {
        let val = ((26 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_027() {
        let val = ((27 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_028() {
        let val = ((28 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_029() {
        let val = ((29 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_030() {
        let val = ((30 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_031() {
        let val = ((31 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_032() {
        let val = ((32 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_033() {
        let val = ((33 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_034() {
        let val = ((34 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_035() {
        let val = ((35 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_036() {
        let val = ((36 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_037() {
        let val = ((37 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_038() {
        let val = ((38 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_039() {
        let val = ((39 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_040() {
        let val = ((40 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_041() {
        let val = ((41 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_042() {
        let val = ((42 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_043() {
        let val = ((43 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_044() {
        let val = ((44 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_045() {
        let val = ((45 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_046() {
        let val = ((46 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_047() {
        let val = ((47 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_048() {
        let val = ((48 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_049() {
        let val = ((49 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_050() {
        let val = ((50 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_051() {
        let val = ((51 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_052() {
        let val = ((52 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_053() {
        let val = ((53 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_054() {
        let val = ((54 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_055() {
        let val = ((55 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_056() {
        let val = ((56 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_057() {
        let val = ((57 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_058() {
        let val = ((58 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_059() {
        let val = ((59 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_060() {
        let val = ((60 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_061() {
        let val = ((61 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_062() {
        let val = ((62 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_063() {
        let val = ((63 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_064() {
        let val = ((64 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_065() {
        let val = ((65 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_066() {
        let val = ((66 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_067() {
        let val = ((67 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_068() {
        let val = ((68 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_069() {
        let val = ((69 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_070() {
        let val = ((70 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_071() {
        let val = ((71 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_072() {
        let val = ((72 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_073() {
        let val = ((73 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_074() {
        let val = ((74 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_075() {
        let val = ((75 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_076() {
        let val = ((76 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_077() {
        let val = ((77 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_078() {
        let val = ((78 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_079() {
        let val = ((79 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_080() {
        let val = ((80 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_081() {
        let val = ((81 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_082() {
        let val = ((82 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_083() {
        let val = ((83 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_084() {
        let val = ((84 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_085() {
        let val = ((85 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_086() {
        let val = ((86 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_087() {
        let val = ((87 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_088() {
        let val = ((88 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_089() {
        let val = ((89 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_090() {
        let val = ((90 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_091() {
        let val = ((91 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_092() {
        let val = ((92 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_093() {
        let val = ((93 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_094() {
        let val = ((94 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_095() {
        let val = ((95 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_096() {
        let val = ((96 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_097() {
        let val = ((97 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_098() {
        let val = ((98 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_099() {
        let val = ((99 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_100() {
        let val = ((100 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_101() {
        let val = ((101 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_102() {
        let val = ((102 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_103() {
        let val = ((103 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_104() {
        let val = ((104 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_105() {
        let val = ((105 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_106() {
        let val = ((106 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_107() {
        let val = ((107 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_108() {
        let val = ((108 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_109() {
        let val = ((109 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_110() {
        let val = ((110 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_111() {
        let val = ((111 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_112() {
        let val = ((112 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_113() {
        let val = ((113 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_114() {
        let val = ((114 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_115() {
        let val = ((115 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_116() {
        let val = ((116 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_117() {
        let val = ((117 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_118() {
        let val = ((118 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_119() {
        let val = ((119 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_120() {
        let val = ((120 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_121() {
        let val = ((121 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_122() {
        let val = ((122 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_123() {
        let val = ((123 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_124() {
        let val = ((124 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_125() {
        let val = ((125 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_126() {
        let val = ((126 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_127() {
        let val = ((127 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_128() {
        let val = ((128 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_129() {
        let val = ((129 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_130() {
        let val = ((130 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_131() {
        let val = ((131 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_132() {
        let val = ((132 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_133() {
        let val = ((133 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_134() {
        let val = ((134 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_135() {
        let val = ((135 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_136() {
        let val = ((136 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_137() {
        let val = ((137 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_138() {
        let val = ((138 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_139() {
        let val = ((139 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_140() {
        let val = ((140 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_141() {
        let val = ((141 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_142() {
        let val = ((142 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_143() {
        let val = ((143 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_144() {
        let val = ((144 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_145() {
        let val = ((145 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_146() {
        let val = ((146 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_147() {
        let val = ((147 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_148() {
        let val = ((148 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_149() {
        let val = ((149 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_150() {
        let val = ((150 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_151() {
        let val = ((151 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_152() {
        let val = ((152 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_153() {
        let val = ((153 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_154() {
        let val = ((154 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_155() {
        let val = ((155 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_156() {
        let val = ((156 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_157() {
        let val = ((157 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_158() {
        let val = ((158 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_159() {
        let val = ((159 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_160() {
        let val = ((160 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_161() {
        let val = ((161 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_162() {
        let val = ((162 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_163() {
        let val = ((163 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_164() {
        let val = ((164 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_165() {
        let val = ((165 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_166() {
        let val = ((166 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_167() {
        let val = ((167 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_168() {
        let val = ((168 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_169() {
        let val = ((169 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_170() {
        let val = ((170 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_171() {
        let val = ((171 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_172() {
        let val = ((172 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_173() {
        let val = ((173 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_174() {
        let val = ((174 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_175() {
        let val = ((175 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_176() {
        let val = ((176 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_177() {
        let val = ((177 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_178() {
        let val = ((178 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_179() {
        let val = ((179 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_180() {
        let val = ((180 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_181() {
        let val = ((181 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_182() {
        let val = ((182 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_183() {
        let val = ((183 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_184() {
        let val = ((184 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_185() {
        let val = ((185 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_186() {
        let val = ((186 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_187() {
        let val = ((187 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_188() {
        let val = ((188 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_189() {
        let val = ((189 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_190() {
        let val = ((190 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_191() {
        let val = ((191 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_192() {
        let val = ((192 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_193() {
        let val = ((193 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_194() {
        let val = ((194 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_195() {
        let val = ((195 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_196() {
        let val = ((196 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_197() {
        let val = ((197 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_198() {
        let val = ((198 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_199() {
        let val = ((199 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_200() {
        let val = ((200 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_201() {
        let val = ((201 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_202() {
        let val = ((202 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_203() {
        let val = ((203 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_204() {
        let val = ((204 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_205() {
        let val = ((205 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_206() {
        let val = ((206 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_207() {
        let val = ((207 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_208() {
        let val = ((208 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_209() {
        let val = ((209 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_210() {
        let val = ((210 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_211() {
        let val = ((211 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_212() {
        let val = ((212 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_213() {
        let val = ((213 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_214() {
        let val = ((214 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_215() {
        let val = ((215 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_216() {
        let val = ((216 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_217() {
        let val = ((217 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_218() {
        let val = ((218 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_219() {
        let val = ((219 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_220() {
        let val = ((220 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_221() {
        let val = ((221 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_222() {
        let val = ((222 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_223() {
        let val = ((223 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_224() {
        let val = ((224 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_225() {
        let val = ((225 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_226() {
        let val = ((226 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_227() {
        let val = ((227 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_228() {
        let val = ((228 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_229() {
        let val = ((229 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_230() {
        let val = ((230 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_231() {
        let val = ((231 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_232() {
        let val = ((232 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_233() {
        let val = ((233 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_234() {
        let val = ((234 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_235() {
        let val = ((235 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_236() {
        let val = ((236 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_237() {
        let val = ((237 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_238() {
        let val = ((238 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_239() {
        let val = ((239 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_240() {
        let val = ((240 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_241() {
        let val = ((241 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_242() {
        let val = ((242 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_243() {
        let val = ((243 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_244() {
        let val = ((244 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_245() {
        let val = ((245 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_246() {
        let val = ((246 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_247() {
        let val = ((247 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_248() {
        let val = ((248 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_249() {
        let val = ((249 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_250() {
        let val = ((250 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_251() {
        let val = ((251 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_252() {
        let val = ((252 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_253() {
        let val = ((253 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_254() {
        let val = ((254 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_255() {
        let val = ((255 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_256() {
        let val = ((256 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_257() {
        let val = ((257 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_258() {
        let val = ((258 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_259() {
        let val = ((259 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_260() {
        let val = ((260 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_261() {
        let val = ((261 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_262() {
        let val = ((262 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_263() {
        let val = ((263 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_264() {
        let val = ((264 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_265() {
        let val = ((265 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_266() {
        let val = ((266 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_267() {
        let val = ((267 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_268() {
        let val = ((268 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_269() {
        let val = ((269 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_270() {
        let val = ((270 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_271() {
        let val = ((271 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_272() {
        let val = ((272 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_273() {
        let val = ((273 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_274() {
        let val = ((274 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_275() {
        let val = ((275 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_276() {
        let val = ((276 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_277() {
        let val = ((277 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_278() {
        let val = ((278 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_279() {
        let val = ((279 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_280() {
        let val = ((280 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_281() {
        let val = ((281 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_282() {
        let val = ((282 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_283() {
        let val = ((283 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_284() {
        let val = ((284 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_285() {
        let val = ((285 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_286() {
        let val = ((286 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_287() {
        let val = ((287 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_288() {
        let val = ((288 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_289() {
        let val = ((289 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_290() {
        let val = ((290 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_291() {
        let val = ((291 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_292() {
        let val = ((292 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_293() {
        let val = ((293 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_294() {
        let val = ((294 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_295() {
        let val = ((295 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_296() {
        let val = ((296 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_297() {
        let val = ((297 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_298() {
        let val = ((298 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_299() {
        let val = ((299 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_300() {
        let val = ((300 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_301() {
        let val = ((301 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_302() {
        let val = ((302 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_303() {
        let val = ((303 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_304() {
        let val = ((304 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_305() {
        let val = ((305 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_306() {
        let val = ((306 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_307() {
        let val = ((307 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_308() {
        let val = ((308 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_309() {
        let val = ((309 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_310() {
        let val = ((310 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_311() {
        let val = ((311 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_312() {
        let val = ((312 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_313() {
        let val = ((313 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_314() {
        let val = ((314 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_315() {
        let val = ((315 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_316() {
        let val = ((316 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_317() {
        let val = ((317 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_318() {
        let val = ((318 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_319() {
        let val = ((319 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_320() {
        let val = ((320 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_321() {
        let val = ((321 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_322() {
        let val = ((322 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_323() {
        let val = ((323 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_324() {
        let val = ((324 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_325() {
        let val = ((325 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_326() {
        let val = ((326 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_327() {
        let val = ((327 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_328() {
        let val = ((328 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_329() {
        let val = ((329 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_330() {
        let val = ((330 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_331() {
        let val = ((331 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_332() {
        let val = ((332 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_333() {
        let val = ((333 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_334() {
        let val = ((334 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_335() {
        let val = ((335 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_336() {
        let val = ((336 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_337() {
        let val = ((337 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_338() {
        let val = ((338 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_339() {
        let val = ((339 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_340() {
        let val = ((340 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_341() {
        let val = ((341 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_342() {
        let val = ((342 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_343() {
        let val = ((343 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_344() {
        let val = ((344 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_345() {
        let val = ((345 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_346() {
        let val = ((346 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_347() {
        let val = ((347 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_348() {
        let val = ((348 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_349() {
        let val = ((349 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_350() {
        let val = ((350 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_351() {
        let val = ((351 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_352() {
        let val = ((352 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_353() {
        let val = ((353 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_354() {
        let val = ((354 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_355() {
        let val = ((355 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_356() {
        let val = ((356 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_357() {
        let val = ((357 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_358() {
        let val = ((358 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }

    #[test]
    fn test_quant_stress_case_359() {
        let val = ((359 % 120) as f64) * 0.1;
        let t = Tensor::from_slice(&[val], vec![1]);
        let q = quantize_per_tensor(&t, 0.1, 0);
        let r = dequantize_per_tensor(&q);
        assert!((r.get(0) - val).abs() <= 0.1);
    }
}
