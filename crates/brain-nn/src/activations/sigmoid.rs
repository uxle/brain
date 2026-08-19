//! # Sigmoid & Hyperbolic Tangent (Tanh)
//!
//! Smooth S-shaped activation functions mapping real values into (0, 1) and (-1, 1).
#![allow(missing_docs)]

use brain_core::Tensor;

/// Computes Sigmoid activation: 1 / (1 + exp(-x)).
pub fn sigmoid(input: &Tensor) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Computes Hyperbolic Tangent (Tanh) activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
pub fn tanh(input: &Tensor) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| x.tanh()).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Sigmoid module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        sigmoid(input)
    }
}

/// Tanh module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

impl Tanh {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        tanh(input)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
