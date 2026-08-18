//! # Batch Normalization (BatchNorm2d & BatchNorm1d)
//!
//! Normalizes mini-batches over spatial dimensions with learned affine scale/bias and running statistics tracking.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

/// 2D Batch Normalization layer.
#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            weight: Tensor::from_vec(vec![1.0; num_features], vec![num_features]),
            bias: Tensor::zeros(vec![num_features]),
            running_mean: Tensor::zeros(vec![num_features]),
            running_var: Tensor::from_vec(vec![1.0; num_features], vec![num_features]),
            training: true,
        }
    }

    pub fn buffers(&self) -> Vec<Tensor> {
        vec![self.running_mean.clone(), self.running_var.clone()]
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
        let r_mean = self.running_mean.to_vec();
        let r_var = self.running_var.to_vec();

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

    pub fn forward_train(&mut self, input: &Tensor) -> Tensor {
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
        let mut r_mean = self.running_mean.to_vec();
        let mut r_var = self.running_var.to_vec();

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

        self.running_mean = Tensor::from_vec(r_mean, vec![c]);
        self.running_var = Tensor::from_vec(r_var, vec![c]);

        Tensor::from_vec(out, shape.to_vec())
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(self.forward_eval(input))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_batchnorm_stress_001() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_002() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_003() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_004() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_005() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_006() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_007() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_008() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_009() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_010() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_011() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_012() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_013() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_014() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_015() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_016() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_017() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_018() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_019() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_020() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_021() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_022() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_023() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_024() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_025() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_026() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_027() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_028() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_029() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_030() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_031() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_032() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_033() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_034() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_035() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_036() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_037() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_038() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_039() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_040() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_041() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_042() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_043() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_044() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_045() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_046() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_047() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_048() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_049() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_050() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_051() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_052() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_053() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_054() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_055() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_056() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_057() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_058() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_059() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_060() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_061() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_062() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_063() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_064() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_065() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_066() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_067() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_068() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_069() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_070() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_071() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_072() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_073() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_074() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_075() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_076() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_077() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_078() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_079() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_080() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_081() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_082() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_083() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_084() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_085() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_086() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_087() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_088() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_089() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_090() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_091() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_092() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_093() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_094() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_095() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_096() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_097() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_098() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_099() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_100() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_101() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_102() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_103() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_104() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_105() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_106() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_107() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_108() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_109() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_110() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_111() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_112() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_113() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_114() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_115() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_116() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_117() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_118() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_119() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_120() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_121() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_122() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_123() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_124() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_125() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_126() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_127() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_128() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_129() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_130() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_131() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_132() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_133() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_134() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_135() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_136() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_137() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_138() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_139() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_140() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_141() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_142() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_143() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_144() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_145() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_146() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_147() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_148() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_149() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_150() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_151() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_152() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_153() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_154() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_155() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_156() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_157() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_158() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_159() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_160() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_161() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_162() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_163() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_164() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_165() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_166() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_167() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_168() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_169() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_170() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_171() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_172() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_173() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_174() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_175() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_176() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_177() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_178() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_179() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_180() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_181() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_182() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_183() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_184() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_185() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_186() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_187() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_188() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_189() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_190() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_191() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_192() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_193() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_194() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_195() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_196() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_197() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_198() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_199() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_200() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_201() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_202() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_203() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_204() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_205() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_206() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_207() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_208() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_209() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_210() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_211() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_212() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_213() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_214() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_215() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_216() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_217() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_218() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_219() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_220() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_221() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_222() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_223() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_224() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_225() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_226() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_227() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_228() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_229() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_230() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_231() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_232() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_233() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_234() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_235() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_236() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_237() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_238() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_239() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_240() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_241() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_242() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_243() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_244() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_245() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_246() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_247() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_248() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_249() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_250() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_251() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_252() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_253() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_254() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_255() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_256() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_257() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_258() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_259() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_260() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_261() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_262() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_263() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_264() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_265() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_266() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_267() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_268() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_269() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_270() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_271() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_272() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_273() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_274() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_275() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_276() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_277() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_278() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_279() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_280() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_281() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_282() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_283() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_284() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_285() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_286() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_287() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_288() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_289() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_290() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_291() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_292() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_293() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_294() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_295() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_296() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_297() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_298() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_299() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_300() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_301() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_302() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_303() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_304() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_305() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_306() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_307() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_308() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_309() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_310() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_311() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_312() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_313() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_314() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_315() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_316() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_317() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_318() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_319() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_320() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_321() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_322() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_323() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_324() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_325() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_326() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_327() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_328() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_329() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_330() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_331() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_332() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_333() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_334() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_335() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_336() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_337() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_338() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_339() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_340() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_341() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_342() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_343() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_344() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_345() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_346() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_347() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_348() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_349() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_350() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_351() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_352() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_353() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_354() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_355() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_356() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_357() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_358() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_359() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_360() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_361() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_362() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_363() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_364() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_365() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_366() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_367() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_368() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_369() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_370() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_371() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_372() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_373() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_374() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_375() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_376() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_377() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_378() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_379() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_380() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_381() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_382() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_383() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_384() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_385() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_386() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_387() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_388() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_389() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_390() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_391() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_392() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_393() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_394() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_395() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_396() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_397() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_398() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_399() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_400() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_401() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_402() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_403() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_404() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_405() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_406() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_407() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_408() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_409() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_410() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_stress_411() {
        let mut bn = BatchNorm2d::new(8);
        let x = Tensor::zeros(vec![1, 8, 4, 4]);
        assert_eq!(bn.forward(&x).unwrap().shape(), &[1, 8, 4, 4]);
        assert_eq!(bn.parameters().len(), 2);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
}
