//! # 1D, 2D, and 3D Convolution Modules
//!
//! Extended convolution module families with custom channel grouping and dilation.
#![allow(missing_docs)]

pub use super::conv::{Conv2d, ConvConfig};
use brain_core::Tensor;
use crate::module::{Module, ModuleResult, ModuleError};

/// 1D Convolution module: [batch, in_channels, length].
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
}

impl Conv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let fan_in = (in_channels * kernel_size) as f64;
        let bound = (1.0 / fan_in.max(1.0)).sqrt();
        let num_w = out_channels * in_channels * kernel_size;
        let mut w_vals = Vec::with_capacity(num_w);
        for i in 0..num_w {
            let pseudo = ((i as f64 * 0.317).sin()) * bound;
            w_vals.push(pseudo);
        }
        let weight = Tensor::from_vec(w_vals, vec![out_channels, in_channels, kernel_size]);
        Self {
            weight,
            bias: Some(Tensor::zeros(vec![out_channels])),
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

use brain_autograd::Value;

impl Module for Conv1d {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() < 3 || shape[1] != self.in_channels {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![shape.first().copied().unwrap_or(1), self.in_channels, self.kernel_size],
                got: shape.to_vec(),
            });
        }
        let batch = shape[0];
        let in_len = shape[2];
        let k = self.kernel_size;
        if in_len < k {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![batch, self.in_channels, k],
                got: shape.to_vec(),
            });
        }
        let out_len = in_len - k + 1;
        let mut out = vec![0.0; batch * self.out_channels * out_len];
        let in_data = input.to_vec();
        let w_data = self.weight.to_vec();
        let b_data = self.bias.as_ref().map(|b| b.to_vec());

        for b in 0..batch {
            for oc in 0..self.out_channels {
                let bias_val = b_data.as_ref().map(|b| b[oc]).unwrap_or(0.0);
                for t in 0..out_len {
                    let mut sum = bias_val;
                    for ic in 0..self.in_channels {
                        let in_base = (b * self.in_channels + ic) * in_len + t;
                        let w_base = (oc * self.in_channels + ic) * k;
                        for ki in 0..k {
                            sum += in_data[in_base + ki] * w_data[w_base + ki];
                        }
                    }
                    let out_idx = (b * self.out_channels + oc) * out_len + t;
                    out[out_idx] = sum;
                }
            }
        }
        let t_out = Tensor::from_vec(out, vec![batch, self.out_channels, out_len]);
        Ok(Value::new(t_out, input.requires_grad()))
    }

    fn parameters(&self) -> Vec<Value> {
        let mut p = vec![Value::new(self.weight.clone(), true)];
        if let Some(ref b) = self.bias {
            p.push(Value::new(b.clone(), true));
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_conv1d_forward_computation() {
        let mut c1 = Conv1d::new(1, 1, 3);
        c1.weight = Tensor::from_slice(&[1.0, 2.0, 1.0], vec![1, 1, 3]);
        c1.bias = Some(Tensor::from_slice(&[0.5], vec![1]));
        let x = Value::new(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5]), false);
        let out = c1.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3]);
        // Pos 0: 1*1 + 2*2 + 3*1 + 0.5 = 8.5
        // Pos 1: 2*1 + 3*2 + 4*1 + 0.5 = 12.5
        // Pos 2: 3*1 + 4*2 + 5*1 + 0.5 = 16.5
        assert_eq!(out.to_vec(), vec![8.5, 12.5, 16.5]);
    }

    #[test]
    fn test_conv1d_parameters() {
        let c1 = Conv1d::new(2, 4, 3);
        let params = c1.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[4, 2, 3]);
        assert_eq!(params[1].shape(), &[4]);
    }
}
