//! # Swish, SiLU & Mish Non-Linearities
//!
//! Self-gated activations: SiLU (Swish), Mish: x * tanh(softplus(x)), and piecewise hard approximations.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Activation kind registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivationKind {
    #[default]
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    FastGELU,
    SiLU,
    Mish,
}

/// Computes SiLU (Swish-1) activation: x * sigmoid(x).
pub fn silu(input: &Tensor) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| {
        let s = 1.0 / (1.0 + (-x).exp());
        x * s
    }).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// Computes Swish activation (alias for SiLU).
pub fn swish(input: &Tensor) -> Tensor {
    silu(input)
}

/// Computes Mish activation: x * tanh(ln(1 + exp(x))).
pub fn mish(input: &Tensor) -> Tensor {
    let data: Vec<f64> = input.to_vec().iter().map(|&x| {
        let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        x * softplus.tanh()
    }).collect();
    Tensor::from_vec(data, input.shape().to_vec())
}

/// SiLU module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct SiLU;

impl SiLU {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        silu(input)
    }
}

/// Swish module wrapper.
pub type Swish = SiLU;

/// Mish module wrapper.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mish;

impl Mish {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        mish(input)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
