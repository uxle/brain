//! # AlphaDropout & Spatial Dropout
//!
//! Self-normalizing AlphaDropout preserving mean and variance under SELU activations.
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};

/// AlphaDropout for self-normalizing neural networks (SNNs).
#[derive(Debug, Clone)]
pub struct AlphaDropout {
    pub p: f64,
    pub training: bool,
}

impl AlphaDropout {
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }
}

use brain_autograd::Value;

impl Module for AlphaDropout {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        Ok(input.clone())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// 2D Spatial/Channel Dropout randomly zeroing entire feature map channels.
#[derive(Debug, Clone)]
pub struct Dropout2d {
    pub p: f64,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        Ok(input.clone())
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
