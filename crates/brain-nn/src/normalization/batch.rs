//! # Batch Normalization (BatchNorm2d & BatchNorm1d)
//!
//! Normalizes mini-batches over spatial dimensions with learned affine scale/bias and running statistics tracking.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};
use std::sync::RwLock;

/// 2D Batch Normalization layer.
#[derive(Debug)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: RwLock<Tensor>,
    pub running_var: RwLock<Tensor>,
    pub training: bool,
}

impl Clone for BatchNorm2d {
    fn clone(&self) -> Self {
        Self {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            running_mean: RwLock::new(self.running_mean.read().unwrap().clone()),
            running_var: RwLock::new(self.running_var.read().unwrap().clone()),
            training: self.training,
        }
    }
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            weight: Tensor::from_vec(vec![1.0; num_features], vec![num_features]),
            bias: Tensor::zeros(vec![num_features]),
            running_mean: RwLock::new(Tensor::zeros(vec![num_features])),
            running_var: RwLock::new(Tensor::from_vec(vec![1.0; num_features], vec![num_features])),
            training: true,
        }
    }

    pub fn buffers(&self) -> Vec<Tensor> {
        vec![
            self.running_mean.read().unwrap().clone(),
            self.running_var.read().unwrap().clone(),
        ]
    }

    pub fn forward_eval(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        if shape.len() < 2 || shape[1] != self.num_features {
            return input.clone();
        }
        let n = shape[0];
        let c = shape[1];
        let spatial_size: usize = shape[2..].iter().product();
        let total = n * c * spatial_size;

        let in_data = input.to_vec();
        let mut out = vec![0.0f64; total];

        let w_data = self.weight.to_vec();
        let b_data = self.bias.to_vec();
        let r_mean = self.running_mean.read().unwrap().to_vec();
        let r_var = self.running_var.read().unwrap().to_vec();

        for b in 0..n {
            for ch in 0..c {
                let mean = r_mean[ch];
                let var = r_var[ch];
                let inv_std = 1.0 / (var + self.eps).sqrt();
                let gamma = w_data[ch];
                let beta = b_data[ch];

                for s in 0..spatial_size {
                    let idx = (b * c + ch) * spatial_size + s;
                    out[idx] = ((in_data[idx] - mean) * inv_std) * gamma + beta;
                }
            }
        }

        Tensor::from_vec(out, shape.to_vec())
    }

    pub fn forward_train(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        if shape.len() < 2 || shape[1] != self.num_features {
            return input.clone();
        }
        let n = shape[0];
        let c = shape[1];
        let spatial_size: usize = shape[2..].iter().product();
        let m = (n * spatial_size) as f64;
        let total = n * c * spatial_size;

        let in_data = input.to_vec();
        let mut out = vec![0.0f64; total];

        let w_data = self.weight.to_vec();
        let b_data = self.bias.to_vec();
        let mut r_mean = self.running_mean.read().unwrap().to_vec();
        let mut r_var = self.running_var.read().unwrap().to_vec();

        for ch in 0..c {
            let mut sum = 0.0;
            for b in 0..n {
                for s in 0..spatial_size {
                    let idx = (b * c + ch) * spatial_size + s;
                    sum += in_data[idx];
                }
            }
            let mean = sum / m.max(1.0);

            let mut var_sum = 0.0;
            for b in 0..n {
                for s in 0..spatial_size {
                    let idx = (b * c + ch) * spatial_size + s;
                    let diff = in_data[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / m.max(1.0);
            let inv_std = 1.0 / (var + self.eps).sqrt();
            let gamma = w_data[ch];
            let beta = b_data[ch];

            for b in 0..n {
                for s in 0..spatial_size {
                    let idx = (b * c + ch) * spatial_size + s;
                    out[idx] = ((in_data[idx] - mean) * inv_std) * gamma + beta;
                }
            }

            r_mean[ch] = (1.0 - self.momentum) * r_mean[ch] + self.momentum * mean;
            r_var[ch] = (1.0 - self.momentum) * r_var[ch] + self.momentum * var;
        }

        *self.running_mean.write().unwrap() = Tensor::from_vec(r_mean, vec![c]);
        *self.running_var.write().unwrap() = Tensor::from_vec(r_var, vec![c]);

        Tensor::from_vec(out, shape.to_vec())
    }
}

use brain_autograd::Value;

impl Module for BatchNorm2d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let t_out = if self.training {
            self.forward_train(input.data())
        } else {
            self.forward_eval(input.data())
        };
        Ok(Value::new(t_out, input.requires_grad()))
    }

    fn parameters(&self) -> Vec<Value> {
        vec![Value::new(self.weight.clone(), true), Value::new(self.bias.clone(), true)]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_batchnorm_train_updates_stats() {
        let bn = BatchNorm2d::new(2);
        let x = Value::new(Tensor::from_vec(vec![10.0, 20.0, 10.0, 20.0], vec![2, 2, 1, 1]), false);
        let out = bn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 2, 1, 1]);

        let r_mean = bn.running_mean.read().unwrap().to_vec();
        assert!(r_mean[0] > 0.0);
        assert!(r_mean[1] > 0.0);
    }

    #[test]
    fn test_batchnorm_eval_mode() {
        let mut bn = BatchNorm2d::new(2);
        bn.set_training(false);
        let x = Value::new(Tensor::from_vec(vec![10.0, 20.0, 10.0, 20.0], vec![2, 2, 1, 1]), false);
        let out = bn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 2, 1, 1]);

        let r_mean = bn.running_mean.read().unwrap().to_vec();
        assert_eq!(r_mean, vec![0.0, 0.0]);
    }
}
