//! # Root Mean Square Layer Normalization (RMSNorm)
//!
//! Parameter-efficient normalization: y = x / RMS(x) * gamma, skipping mean centering for transformer stacks.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

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
    pub weight: Tensor,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            dim,
            eps,
            weight: Tensor::from_vec(vec![1.0; dim], vec![dim]),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let total: usize = shape.iter().product();
        let batch_items = total / self.dim.max(1);

        let data = input.to_vec();
        let w_data = self.weight.to_vec();
        let mut out = vec![0.0f64; total];

        for b in 0..batch_items {
            let slice = &data[b * self.dim..(b + 1) * self.dim];
            let mean_sq: f64 = slice.iter().map(|&x| x * x).sum::<f64>() / self.dim as f64;
            let rms = 1.0 / (mean_sq + self.eps).sqrt();

            for i in 0..self.dim {
                out[b * self.dim + i] = slice[i] * rms * w_data[i];
            }
        }

        Tensor::from_vec(out, shape.to_vec())
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(self.forward(input))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
