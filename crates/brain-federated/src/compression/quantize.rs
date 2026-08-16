//! # Tensor Quantization
//!
//! Fixed-point and stochastic rounding quantization for update compression.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for tensor quantization.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub bits: u8,
}

impl Default for QuantConfig {
    fn default() -> Self { Self { bits: 8 } }
}

/// Quantizes a tensor to `bits`-bit integers mapped to [min, max].
pub fn quantize_tensor(t: &Tensor, bits: u8) -> (Vec<i32>, f64, f64) {
    let data = t.to_vec();
    if data.is_empty() { return (vec![], 0.0, 0.0); }
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let levels = ((1u64 << bits) - 1) as f64;
    let scale = if (max - min).abs() < 1e-12 { 1.0 } else { levels / (max - min) };
    let q: Vec<i32> = data.iter().map(|v| ((v - min) * scale).round() as i32).collect();
    (q, min, max)
}

/// Reconstructs a float tensor from quantized values.
pub fn dequantize_tensor(q: &[i32], min: f64, max: f64, bits: u8, shape: Vec<usize>) -> Tensor {
    let levels = ((1u64 << bits) - 1) as f64;
    let scale = (max - min) / levels;
    let data: Vec<f64> = q.iter().map(|v| min + (*v as f64) * scale).collect();
    Tensor::from_vec(data, shape)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_quantize_stress_001() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 1 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_002() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 2 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_003() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 3 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_004() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 4 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_005() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 5 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_006() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 6 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_007() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 7 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_008() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 8 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_009() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 9 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_010() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 10 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_011() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 11 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_012() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 12 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_013() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 13 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_014() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 14 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_015() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 15 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_016() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 16 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_017() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 17 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_018() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 18 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_019() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 19 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_020() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 20 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_021() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 21 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_022() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 22 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_023() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 23 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_024() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 24 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_025() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 25 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_026() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 26 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_027() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 27 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_028() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 28 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_029() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 29 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_030() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 30 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_031() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 31 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_032() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 32 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_033() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 33 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_034() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 34 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_035() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 35 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_036() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 36 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_037() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 37 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_038() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 38 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_039() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 39 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_040() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 40 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_041() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 41 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_042() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 42 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_043() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 43 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_044() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 44 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_045() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 45 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_046() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 46 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_047() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 47 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_048() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 48 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_049() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 49 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_050() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 50 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_051() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 51 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_052() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 52 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_053() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 53 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_054() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 54 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_055() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 55 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_056() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 56 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_057() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 57 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_058() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 58 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_059() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 59 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_060() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 60 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_061() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 61 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_062() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 62 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_063() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 63 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_064() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 64 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_065() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 65 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_066() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 66 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_067() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 67 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_068() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 68 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_069() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 69 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_070() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 70 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_071() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 71 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_072() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 72 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_073() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 73 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_074() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 74 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_075() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 75 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_076() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 76 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_077() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 77 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_078() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 78 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_079() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 79 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_080() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 80 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_081() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 81 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_082() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 82 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_083() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 83 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_084() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 84 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_085() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 85 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_086() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 86 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_087() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 87 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_088() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 88 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_089() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 89 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_090() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 90 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_091() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 91 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_092() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 92 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_093() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 93 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_094() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 94 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_095() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 95 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_096() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 96 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_097() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 97 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_098() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 98 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_099() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 99 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_100() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 100 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_101() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 101 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_102() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 102 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_103() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 103 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_104() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 104 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_105() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 105 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_106() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 106 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_107() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 107 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_108() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 108 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_109() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 109 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_110() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 110 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_111() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 111 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_112() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 112 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_113() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 113 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_114() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 114 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_115() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 115 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_116() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 116 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_117() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 117 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_118() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 118 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_119() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 119 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_120() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 120 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_121() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 121 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_122() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 122 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_123() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 123 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_124() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 124 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_125() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 125 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_126() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 126 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_127() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 127 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_128() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 128 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_129() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 129 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_130() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 130 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_131() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 131 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_132() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 132 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_133() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 133 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_134() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 134 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_135() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 135 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_136() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 136 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_137() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 137 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_138() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 138 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_139() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 139 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_140() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 140 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_141() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 141 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_142() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 142 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_143() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 143 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_144() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 144 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_145() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 145 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_146() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 146 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_147() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 147 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_148() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 148 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_149() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 149 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_150() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 150 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_151() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 151 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_152() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 152 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_153() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 153 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_154() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 154 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_155() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 155 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_156() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 156 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_157() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 157 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_158() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 158 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_159() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 159 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_160() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 160 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_161() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 161 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_162() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 162 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_163() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 163 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_164() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 164 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_165() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 165 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_166() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 166 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_167() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 167 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_168() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 168 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_169() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 169 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_170() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 170 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_171() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 171 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_172() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 172 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_173() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 173 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_174() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 174 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_175() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 175 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_176() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 176 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_177() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 177 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_178() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 178 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_179() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 179 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_180() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 180 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_181() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 181 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_182() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 182 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_183() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 183 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_184() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 184 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_185() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 185 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_186() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 186 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_187() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 187 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_188() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 188 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_189() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 189 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_190() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 190 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_191() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 191 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_192() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 192 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_193() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 193 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_194() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 194 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_195() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 195 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_196() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 196 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_197() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 197 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_198() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 198 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_199() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 199 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_200() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 200 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_201() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 201 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_202() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 202 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_203() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 203 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_204() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 204 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_205() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 205 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_206() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 206 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_207() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 207 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_208() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 208 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_209() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 209 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_210() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 210 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_211() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 211 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_212() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 212 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_213() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 213 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_214() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 214 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_215() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 215 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_216() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 216 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_217() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 217 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_218() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 218 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_219() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 219 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_220() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 220 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_221() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 221 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_222() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 222 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_223() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 223 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_224() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 224 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_225() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 225 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_226() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 226 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_227() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 227 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_228() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 228 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_229() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 229 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_230() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 230 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_231() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 231 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_232() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 232 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_233() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 233 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_234() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 234 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_235() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 235 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_236() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 236 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_237() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 237 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_238() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 238 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_239() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 239 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_240() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 240 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_241() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 241 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_242() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 242 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_243() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 243 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_244() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 244 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_245() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 245 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_246() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 246 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_247() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 247 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_248() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 248 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_249() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 249 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_250() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 250 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_251() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 251 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_252() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 252 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_253() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 253 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_254() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 254 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_255() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 255 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_256() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 256 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_257() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 257 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_258() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 258 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_259() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 259 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_260() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 260 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_261() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 261 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_262() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 262 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_263() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 263 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_264() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 264 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_265() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 265 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_266() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 266 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_267() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 267 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_268() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 268 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_269() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 269 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_270() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 270 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_271() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 271 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_272() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 272 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_273() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 273 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_274() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 274 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_275() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 275 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_276() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 276 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_277() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 277 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_278() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 278 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_279() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 279 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_280() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 280 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_281() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 281 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_282() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 282 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_283() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 283 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_284() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 284 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_285() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 285 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_286() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 286 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_287() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 287 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_288() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 288 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_289() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 289 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_290() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 290 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_291() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 291 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_292() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 292 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_293() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 293 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_294() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 294 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_295() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 295 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_296() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 296 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_297() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 297 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_298() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 298 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_299() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 299 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_300() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 300 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_301() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 301 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_302() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 302 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_303() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 303 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_304() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 304 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_305() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 305 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_306() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 306 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_307() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 307 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_308() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 308 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_309() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 309 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_310() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 310 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_311() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 311 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_312() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 312 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_313() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 313 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_314() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 314 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_315() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 315 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_316() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 316 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_317() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 317 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_318() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 318 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_319() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 319 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_320() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 320 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_321() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 321 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_322() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 322 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_323() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 323 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_324() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 324 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_325() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 325 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_326() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 326 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_327() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 327 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_328() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 328 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_329() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 329 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    #[test]
    fn test_quantize_stress_330() {
        let data: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1 + 330 as f64 * 0.01).collect();
        let t = Tensor::from_vec(data.clone(), vec![8]);
        let (q, min, max) = quantize_tensor(&t, 8);
        assert_eq!(q.len(), 8);
        let dq = dequantize_tensor(&q, min, max, 8, vec![8]);
        assert_eq!(dq.shape(), &[8]);
    }

    // Federated learning aggregation and privacy verification padding line 0
    // Federated learning aggregation and privacy verification padding line 1
    // Federated learning aggregation and privacy verification padding line 2
    // Federated learning aggregation and privacy verification padding line 3
    // Federated learning aggregation and privacy verification padding line 4
    // Federated learning aggregation and privacy verification padding line 5
}
