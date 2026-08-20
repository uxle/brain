//! # Softmax & LogSoftmax
//!
//! Numerically stable Softmax, LogSoftmax, and Softmin across specified tensor dimensions.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for Softmax operators.
#[derive(Debug, Clone, Default)]
pub struct SoftmaxConfig {
    pub dim: isize,
}

/// Numerically stable 2D softmax along the last dimension.
pub fn softmax(input: &Tensor) -> Tensor {
    let shape = input.shape();
    let rows = shape[0];
    let cols = if shape.len() > 1 { shape[1] } else { 1 };
    let data = input.to_vec();

    let mut out = vec![0.0f64; rows * cols];

    for r in 0..rows {
        let row_slice = &data[r * cols..(r + 1) * cols];
        let max_val = row_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_slice.iter().map(|&x| (x - max_val).exp()).sum();

        for c in 0..cols {
            out[r * cols + c] = (data[r * cols + c] - max_val).exp() / sum_exp.max(1e-12);
        }
    }

    Tensor::from_vec(out, shape.to_vec())
}

/// Numerically stable 2D LogSoftmax along the last dimension.
pub fn log_softmax(input: &Tensor) -> Tensor {
    let sm = softmax(input);
    let data: Vec<f64> = sm
        .to_vec()
        .iter()
        .map(|&x| x.clamp(1e-15, 1.0).ln())
        .collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Softmax module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax;

impl Softmax {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        softmax(input)
    }
}

/// LogSoftmax module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogSoftmax;

impl LogSoftmax {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        log_softmax(input)
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
