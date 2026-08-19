//! # Instance Normalization (InstanceNorm2d)
//!
//! Normalizes each sample-channel independently over spatial dimensions, with
//! optional learned affine scale/bias (torch `InstanceNorm2d` semantics).
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// 2D Instance Normalization layer.
#[derive(Debug, Clone)]
pub struct InstanceNorm2d {
    pub num_features: usize,
    pub eps: f64,
    pub affine: bool,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl InstanceNorm2d {
    pub fn new(num_features: usize, affine: bool) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            affine,
            weight: Tensor::from_vec(vec![1.0; num_features], vec![num_features]),
            bias: Tensor::zeros(vec![num_features]),
        }
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let shape = input.shape();
        if shape.len() < 4 || shape[1] != self.num_features {
            return Err(crate::module::ModuleError::ShapeMismatch {
                expected: vec![shape.first().copied().unwrap_or(1), self.num_features, 1, 1],
                got: shape.to_vec(),
            });
        }
        let n = shape[0];
        let c = shape[1];
        let spatial_size: usize = shape[2..].iter().product();
        let total = n * c * spatial_size;

        let in_data = input.to_vec();
        let mut out = vec![0.0f64; total];

        let w_data = if self.affine { self.weight.to_vec() } else { vec![1.0; c] };
        let b_data = if self.affine { self.bias.to_vec() } else { vec![0.0; c] };

        for b in 0..n {
            for ch in 0..c {
                let start = (b * c + ch) * spatial_size;
                let slice = &in_data[start..start + spatial_size];
                let mean: f64 = slice.iter().sum::<f64>() / (spatial_size as f64);
                let var: f64 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (spatial_size as f64);
                let inv_std = 1.0 / (var + self.eps).sqrt();
                let gamma = w_data[ch];
                let beta = b_data[ch];

                for s in 0..spatial_size {
                    out[start + s] = ((in_data[start + s] - mean) * inv_std) * gamma + beta;
                }
            }
        }

        Ok(Tensor::from_vec(out, shape.to_vec()))
    }

    fn parameters(&self) -> Vec<Tensor> {
        if self.affine {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_norm2d_no_affine() {
        let in_ = InstanceNorm2d::new(2, false);
        let t = Tensor::from_slice(
            &[1.0, 3.0, 2.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            vec![1, 2, 2, 2],
        );
        let out = in_.forward(&t).unwrap();
        // Channel 0: mean 2.5, pop-var 1.25 -> inv_std ~0.8944
        let inv = 1.0 / (1.25f64 + 1e-5).sqrt();
        assert!((out.get(0) - (1.0 - 2.5) * inv).abs() < 1e-6);
        // Channel 1: mean 25, pop-var 125 -> each element normalized
        let inv2 = 1.0 / (125.0f64 + 1e-5).sqrt();
        assert!((out.get(4) - (10.0 - 25.0) * inv2).abs() < 1e-6);
        // Parameters empty without affine
        assert!(in_.parameters().is_empty());
    }

    #[test]
    fn test_instance_norm2d_affine() {
        let in_ = InstanceNorm2d::new(1, true);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = in_.forward(&t).unwrap();
        // mean 2.5, var 1.25, gamma 1, beta 0
        let inv = 1.0 / (1.25f64 + 1e-5).sqrt();
        assert!((out.get(0) - (1.0 - 2.5) * inv).abs() < 1e-6);
        assert_eq!(in_.parameters().len(), 2);
    }

    #[test]
    fn test_instance_norm_per_sample_independent() {
        let in_ = InstanceNorm2d::new(1, false);
        let t = Tensor::from_slice(&[1.0, 3.0, 100.0, 300.0], vec![2, 1, 1, 2]);
        let out = in_.forward(&t).unwrap();
        // Both samples normalize to the same values: [-1, 1] (within eps tolerance)
        assert!((out.get(0) + 1.0).abs() < 1e-4);
        assert!((out.get(1) - 1.0).abs() < 1e-4);
        assert!((out.get(2) + 1.0).abs() < 1e-4);
        assert!((out.get(3) - 1.0).abs() < 1e-4);
    }
}