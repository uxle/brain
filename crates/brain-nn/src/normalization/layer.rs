//! # Layer Normalization (LayerNorm)
//!
//! Standard Layer Normalization over specified normalized shapes.
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};
use brain_core::Tensor;

/// Layer Normalization module.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,
    pub eps: f64,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Self {
        let size: usize = normalized_shape.iter().product();
        Self {
            normalized_shape: normalized_shape.clone(),
            eps,
            weight: Tensor::from_vec(vec![1.0; size], normalized_shape.clone()),
            bias: Tensor::zeros(normalized_shape),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let norm_size: usize = self.normalized_shape.iter().product();
        let total: usize = shape.iter().product();
        let batch_items = total / norm_size.max(1);

        let data = input.to_vec();
        let w_data = self.weight.to_vec();
        let b_data = self.bias.to_vec();

        let mut out = vec![0.0f64; total];

        for b in 0..batch_items {
            let slice = &data[b * norm_size..(b + 1) * norm_size];
            let mean: f64 = slice.iter().sum::<f64>() / norm_size as f64;
            let var: f64 =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / norm_size as f64;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for i in 0..norm_size {
                let norm = (slice[i] - mean) * inv_std;
                out[b * norm_size + i] = norm * w_data[i] + b_data[i];
            }
        }

        Tensor::from_vec(out, shape.to_vec())
    }
}

use brain_autograd::Value;

impl Module for LayerNorm {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let t_out = self.forward(input.data());
        Ok(Value::new(t_out, input.requires_grad()))
    }

    fn parameters(&self) -> Vec<Value> {
        vec![
            Value::new(self.weight.clone(), true),
            Value::new(self.bias.clone(), true),
        ]
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
