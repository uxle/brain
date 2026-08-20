//! # Standard Inverted Dropout
//!
//! Randomly zeroes elements with probability p during training, scaling non-zero entries by 1/(1-p).
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};
use brain_core::Tensor;

/// Standard inverted Bernoulli dropout.
#[derive(Debug, Clone)]
pub struct Dropout {
    pub p: f64,
    pub training: bool,
    pub seed: u64,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self {
            p,
            training: true,
            seed: 12345,
        }
    }

    pub fn with_seed(p: f64, seed: u64) -> Self {
        Self {
            p,
            training: true,
            seed,
        }
    }

    pub fn forward_tensor(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p <= 0.0 {
            return input.clone();
        }

        let scale = 1.0 / (1.0 - self.p);
        let total: usize = input.shape().iter().product();
        let data = input.to_vec();

        let mut out = Vec::with_capacity(total);
        for (i, &val) in data.iter().enumerate() {
            let rnd = ((i as u64 + self.seed) * 1103515245 + 12345) % 65536;
            let prob = rnd as f64 / 65536.0;
            if prob >= self.p {
                out.push(val * scale);
            } else {
                out.push(0.0);
            }
        }

        Tensor::from_vec(out, input.shape().to_vec())
    }
}

use brain_autograd::Value;

impl Module for Dropout {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let t_out = self.forward_tensor(input.data());
        Ok(Value::new(t_out, input.requires_grad()))
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Fused Dropout + Residual Addition module.
#[derive(Debug, Clone)]
pub struct FusedDropout {
    pub dropout: Dropout,
}

impl FusedDropout {
    pub fn new(p: f64) -> Self {
        Self {
            dropout: Dropout::new(p),
        }
    }

    pub fn forward_add(&self, input: &Tensor, residual: &Tensor) -> Tensor {
        let dropped = self.dropout.forward_tensor(input);
        &dropped + residual
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
