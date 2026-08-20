//! # Group Normalization & Instance Normalization
//!
//! Normalization dividing channels into groups, independent of mini-batch size.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// Group Normalization module.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f64,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            weight: Tensor::from_vec(vec![1.0; num_channels], vec![num_channels]),
            bias: Tensor::zeros(vec![num_channels]),
        }
    }
}

use brain_autograd::Value;

impl Module for GroupNorm {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<Value> {
        vec![Value::new(self.weight.clone(), true), Value::new(self.bias.clone(), true)]
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
