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
}
