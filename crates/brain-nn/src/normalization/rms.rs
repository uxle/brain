//! # Root Mean Square Layer Normalization (RMSNorm)
//!
//! Parameter-efficient normalization: y = x / RMS(x) * gamma, skipping mean centering for transformer stacks.
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};
use brain_autograd::Value;
use brain_core::Tensor;

/// Configuration for RMSNorm.
#[derive(Debug, Clone, Default)]
pub struct RMSNormConfig {
    pub dim: usize,
    pub eps: f64,
}

/// Root Mean Square Layer Normalization module.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub dim: usize,
    pub eps: f64,
    pub weight: Value,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            dim,
            eps,
            weight: Value::new(Tensor::from_vec(vec![1.0; dim], vec![dim]), true),
        }
    }

    pub fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        let last_dim = *shape.last().unwrap_or(&0);
        if last_dim != self.dim {
            return Err(crate::module::ModuleError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![last_dim],
            });
        }

        let n = shape.iter().take(shape.len() - 1).product::<usize>().max(1);
        let x_2d = input.reshape(vec![n, self.dim]);

        // x^2
        let x_sq = &x_2d * &x_2d;
        // mean(x^2) across feature dim via matmul with [dim, 1] vector of 1/dim
        let scale_vec = Value::new(
            Tensor::from_vec(vec![1.0 / (self.dim as f64); self.dim], vec![self.dim, 1]),
            false,
        );
        let mean_sq = x_sq.matmul(&scale_vec);
        // rms = (mean_sq + eps)^(-0.5)
        let eps_val = Value::scalar(self.eps);
        let rsqrt_rms = (&mean_sq + &eps_val).pow(&Value::scalar(-0.5));

        // x * rsqrt_rms
        let normalized = &x_2d * &rsqrt_rms;
        // normalized * weight
        let out_2d = &normalized * &self.weight;

        Ok(out_2d.reshape(shape.to_vec()))
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Value> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_rmsnorm_forward_and_backward() {
        let norm = RMSNorm::new(4, 1e-5);
        let x = Value::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0], vec![2, 4]),
            true,
        );
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 4]);

        let loss = y.sum();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert!(norm.weight.grad().is_some());
    }
}
