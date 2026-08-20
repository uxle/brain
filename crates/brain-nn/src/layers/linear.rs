//! # Fully Connected (Dense / Linear) Layer
//!
//! y = x * W^T + b with Kaiming uniform weight initialization and optional bias.
#![allow(missing_docs)]

use brain_core::Tensor;
use brain_autograd::Value;
use crate::module::{Module, ModuleResult, ModuleError};
use crate::init::kaiming_uniform;

/// Linear / Dense transformation layer.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Value,
    pub bias: Option<Value>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        let weight_t = kaiming_uniform(&[out_features, in_features], 0.0);
        let weight = Value::new(weight_t, true);
        let bias = if has_bias {
            Some(Value::new(Tensor::zeros(vec![out_features]), true))
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

    pub fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        let last_dim = *shape.last().unwrap_or(&0);
        if last_dim != self.in_features {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![last_dim],
            });
        }
        Ok(input.linear(&self.weight, self.bias.as_ref()))
    }
}

impl Module for Linear {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Value> {
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

    #[test]
    fn test_linear_weight_and_bias_gradient_via_real_tape() {
        let mut linear = Linear::new(3, 2, true);
        linear.weight = Value::new(
            Tensor::from_slice(&[0.5, -0.2, 1.0, 0.8, -1.2, 0.3], vec![2, 3]),
            true,
        );
        linear.bias = Some(Value::new(Tensor::from_slice(&[0.1, -0.4], vec![2]), true));

        let x = Value::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, -1.0, 0.5, 2.0], vec![2, 3]),
            false,
        );

        let out = linear.forward(&x).unwrap();
        let loss = out.sum();
        loss.backward().unwrap();

        let x_data = x.data().to_vec();
        let mut analytic_dw = vec![0.0f64; 6];
        for o in 0..2 {
            for i in 0..3 {
                analytic_dw[o * 3 + i] = x_data[i] + x_data[3 + i];
            }
        }
        let analytic_db = [2.0, 2.0];

        let w_grad = linear.weight.grad().expect("weight grad should be populated").to_vec();
        let b_grad = linear.bias.as_ref().unwrap().grad().expect("bias grad should be populated").to_vec();

        for i in 0..6 {
            assert!(
                (w_grad[i] - analytic_dw[i]).abs() < 1e-9,
                "weight grad mismatch at {i}: tape={}, expected={}",
                w_grad[i],
                analytic_dw[i]
            );
        }
        for i in 0..2 {
            assert!(
                (b_grad[i] - analytic_db[i]).abs() < 1e-9,
                "bias grad mismatch at {i}: tape={}, expected={}",
                b_grad[i],
                analytic_db[i]
            );
        }
    }

    #[test]
    fn test_linear_shape_mismatch_still_errors() {
        let linear = Linear::new(3, 2, true);
        let bad_input = Value::new(Tensor::from_slice(&[1.0, 2.0], vec![1, 2]), false);
        let result = linear.forward(&bad_input);
        assert!(matches!(result, Err(ModuleError::ShapeMismatch { .. })));
    }
}
