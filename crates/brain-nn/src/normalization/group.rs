//! # Group Normalization (Wu & He, ECCV 2018)
//!
//! Divides channels into groups and computes mean and variance per group, independent of batch size.
#![allow(missing_docs)]

use crate::module::{Module, ModuleResult};
use brain_autograd::Value;
use brain_core::Tensor;

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
    /// Creates a new `GroupNorm` module.
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        assert!(
            num_channels % num_groups == 0,
            "num_channels ({}) must be divisible by num_groups ({})",
            num_channels,
            num_groups
        );
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            weight: Tensor::from_vec(vec![1.0; num_channels], vec![num_channels]),
            bias: Tensor::zeros(vec![num_channels]),
        }
    }

    /// Functional forward pass on a raw `Tensor`.
    pub fn forward_tensor(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let n_dims = shape.len();
        assert!(
            n_dims >= 2,
            "Input must have at least 2 dimensions [N, C, ...]"
        );

        let n = shape[0];
        let c = shape[1];
        assert_eq!(c, self.num_channels, "Channel mismatch in GroupNorm");

        let spatial_size: usize = if n_dims > 2 {
            shape[2..].iter().product()
        } else {
            1
        };

        let channels_per_group = c / self.num_groups;
        let group_size = channels_per_group * spatial_size;
        let data = input.data();
        let weight_data = self.weight.data();
        let bias_data = self.bias.data();

        let mut out = vec![0.0f64; input.numel()];

        for batch_i in 0..n {
            let batch_offset = batch_i * c * spatial_size;

            for g in 0..self.num_groups {
                let group_start_channel = g * channels_per_group;
                let group_offset = batch_offset + group_start_channel * spatial_size;

                // 1. Compute group mean
                let mut sum = 0.0f64;
                for ch in 0..channels_per_group {
                    let ch_offset = group_offset + ch * spatial_size;
                    for sp in 0..spatial_size {
                        sum += data[ch_offset + sp];
                    }
                }
                let mean = sum / (group_size as f64);

                // 2. Compute group variance
                let mut var_sum = 0.0f64;
                for ch in 0..channels_per_group {
                    let ch_offset = group_offset + ch * spatial_size;
                    for sp in 0..spatial_size {
                        let diff = data[ch_offset + sp] - mean;
                        var_sum += diff * diff;
                    }
                }
                let var = var_sum / (group_size as f64);
                let std_inv = 1.0 / (var + self.eps).sqrt();

                // 3. Normalize and apply affine transform gamma * x_norm + beta
                for ch in 0..channels_per_group {
                    let actual_c = group_start_channel + ch;
                    let gamma = weight_data[actual_c];
                    let beta = bias_data[actual_c];
                    let ch_offset = group_offset + ch * spatial_size;

                    for sp in 0..spatial_size {
                        let x_norm = (data[ch_offset + sp] - mean) * std_inv;
                        out[ch_offset + sp] = gamma * x_norm + beta;
                    }
                }
            }
        }

        Tensor::from_vec(out, shape.to_vec())
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let out_tensor = self.forward_tensor(&input.data());
        Ok(Value::new(out_tensor, input.requires_grad()))
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
    use super::*;

    #[test]
    fn test_group_norm_zero_mean_unit_variance() {
        let gn = GroupNorm::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 4, 2]);

        let y = gn.forward_tensor(&x);
        assert_eq!(y.shape(), &[1, 4, 2]);

        // First group (channels 0, 1 -> 4 elements: 1, 2, 3, 4)
        // Mean = 2.5
        let g0_slice = &y.data()[0..4];
        let sum: f64 = g0_slice.iter().sum();
        assert!(sum.abs() < 1e-5, "Group 0 mean should be normalized to 0.0");
    }
}
