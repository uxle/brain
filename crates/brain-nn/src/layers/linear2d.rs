//! # Bilinear & Identity Layers
//!
//! Bilinear transformation y = x1 * W * x2^T + b and parameter-free Identity pass-through.
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};
use brain_core::Tensor;

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
    pub weight: Value,
    pub bias: Option<Value>,
}

impl Bilinear {
    pub fn new(in1: usize, in2: usize, out: usize, has_bias: bool) -> Self {
        let weight_t = crate::init::kaiming_uniform(&[out, in1, in2], 0.0);
        let weight = Value::new(weight_t, true);
        let bias = if has_bias {
            Some(Value::new(Tensor::zeros(vec![out]), true))
        } else {
            None
        };
        Self {
            in1_features: in1,
            in2_features: in2,
            out_features: out,
            weight,
            bias,
        }
    }

    /// Bilinear forward pass: input1 [N, in1], input2 [N, in2] -> [N, out].
    pub fn forward(&self, input1: &Value, input2: &Value) -> ModuleResult<Value> {
        let s1 = input1.shape();
        let s2 = input2.shape();
        if s1.len() != 2 || s2.len() != 2 {
            return Err(crate::module::ModuleError::InvalidParameter(
                "Bilinear inputs must be 2D tensors [batch_size, features]".to_string(),
            ));
        }
        if s1[0] != s2[0] {
            return Err(crate::module::ModuleError::ShapeMismatch {
                expected: vec![s1[0], self.in2_features],
                got: s2.to_vec(),
            });
        }
        if s1[1] != self.in1_features || s2[1] != self.in2_features {
            return Err(crate::module::ModuleError::ShapeMismatch {
                expected: vec![self.in1_features, self.in2_features],
                got: vec![s1[1], s2[1]],
            });
        }

        let n = s1[0];
        let out_feat = self.out_features;
        let in1 = self.in1_features;
        let in2 = self.in2_features;

        let ones_in2 = Value::new(Tensor::ones(vec![in2, 1]), false);

        let mut out_cols = Vec::with_capacity(out_feat);
        for k in 0..out_feat {
            let w_k_data = Tensor::from_vec(
                (0..in1)
                    .flat_map(|i| (0..in2).map(move |j| self.weight.data().get_3d(k, i, j)))
                    .collect(),
                vec![in1, in2],
            );
            let w_k = Value::new(w_k_data, self.weight.requires_grad());
            let m_k = input1.matmul(&w_k);
            let col_k = (&m_k * input2).matmul(&ones_in2);
            out_cols.push(col_k);
        }

        let mut out_t = Tensor::zeros(vec![n, out_feat]);
        for (k, col) in out_cols.iter().enumerate() {
            for i in 0..n {
                out_t.set_2d(i, k, col.data().get_2d(i, 0));
            }
        }
        let mut result = Value::new(
            out_t,
            input1.requires_grad() || input2.requires_grad() || self.weight.requires_grad(),
        );
        if let Some(ref b) = self.bias {
            result = &result + b;
        }
        Ok(result)
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_bilinear_forward_and_params() {
        let bilinear = Bilinear::new(3, 4, 2, true);
        assert_eq!(bilinear.parameters().len(), 2);
        let x1 = Value::new(Tensor::ones(vec![2, 3]), false);
        let x2 = Value::new(Tensor::ones(vec![2, 4]), false);
        let y = bilinear.forward(&x1, &x2).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
    }
}
