//! # Fully Connected (Dense / Linear) Layer
//!
//! y = x * W^T + b with Kaiming uniform weight initialization and optional bias.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult, ModuleError};
use crate::init::kaiming_uniform;

/// Linear / Dense transformation layer.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        let weight = kaiming_uniform(&[out_features, in_features], 0.0);
        let bias = if has_bias {
            Some(Tensor::zeros(vec![out_features]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let shape = input.shape();
        let last_dim = *shape.last().unwrap_or(&0);
        if last_dim != self.in_features {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![last_dim],
            });
        }

        let total: usize = shape.iter().product();
        let num_vectors = total / self.in_features.max(1);
        let in_data = input.to_vec();
        let w_data = self.weight.to_vec();

        let mut out_data = vec![0.0f64; num_vectors * self.out_features];
        for b in 0..num_vectors {
            for o in 0..self.out_features {
                let mut sum = if let Some(ref bias) = self.bias { bias.to_vec()[o] } else { 0.0 };
                for i in 0..self.in_features {
                    sum += in_data[b * self.in_features + i] * w_data[o * self.in_features + i];
                }
                out_data[b * self.out_features + o] = sum;
            }
        }

        let mut out_shape = shape.to_vec();
        if let Some(last) = out_shape.last_mut() {
            *last = self.out_features;
        }
        Ok(Tensor::from_vec(out_data, out_shape))
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
