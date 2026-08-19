//! # Gaussian Error Linear Unit (GELU)
//!
//! GELU activation using exact error function (erf) and fast tanh polynomial approximations.
#![allow(missing_docs)]

use brain_core::Tensor;
use std::f64::consts::PI;

/// Computes exact/approximate GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
pub fn gelu(input: &Tensor) -> Tensor {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    let data: Vec<f64> = input.to_vec().iter().map(|&x| {
        let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
        0.5 * x * (1.0 + inner.tanh())
    }).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Fast GELU approximation: x * sigmoid(1.702 * x).
pub fn fast_gelu(input: &Tensor) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| {
        let s = 1.0 / (1.0 + (-1.702 * x).exp());
        x * s
    }).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// GELU module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct GELU;

impl GELU {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        gelu(input)
    }
}

/// FastGELU module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct FastGELU;

impl FastGELU {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        fast_gelu(input)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
