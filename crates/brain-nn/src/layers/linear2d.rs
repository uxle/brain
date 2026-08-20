//! # Bilinear & Identity Layers
//!
//! Bilinear transformation y = x1 * W * x2^T + b and parameter-free Identity pass-through.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

use brain_autograd::Value;

/// Identity pass-through module.
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

impl Identity {
    pub fn forward(&self, input: &Value) -> ModuleResult<Value> {
        Ok(input.clone())
    }
}

impl Module for Identity {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        self.forward(input)
    }
}

/// Bilinear transformation module: y = x1 * W * x2 + b.
#[derive(Debug, Clone)]
pub struct Bilinear {
    pub in1_features: usize,
    pub in2_features: usize,
    pub out_features: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Bilinear {
    pub fn new(in1: usize, in2: usize, out: usize, has_bias: bool) -> Self {
        let weight = Tensor::zeros(vec![out, in1, in2]);
        let bias = if has_bias { Some(Tensor::zeros(vec![out])) } else { None };
        Self {
            in1_features: in1,
            in2_features: in2,
            out_features: out,
            weight,
            bias,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
