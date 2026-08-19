//! # Rectified Linear Units (ReLU & LeakyReLU)
//!
//! Standard rectified linear units and parameterized leaky rectifiers.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Computes standard ReLU activation: max(0, x).
pub fn relu(input: &Tensor) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| x.max(0.0)).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Computes LeakyReLU activation: max(negative_slope * x, x).
pub fn leaky_relu(input: &Tensor, negative_slope: f64) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| if x >= 0.0 { x } else { negative_slope * x }).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// ReLU module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;

impl ReLU {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        relu(input)
    }
}

/// LeakyReLU module wrapper.
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    pub negative_slope: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self { negative_slope: 0.01 }
    }
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        leaky_relu(input, self.negative_slope)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
