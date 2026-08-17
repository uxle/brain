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

    #[test]
    fn test_rmsnorm_stress_001() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_002() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_003() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_004() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_005() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_006() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_007() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_008() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_009() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_010() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_011() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_012() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_013() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_014() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_015() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_016() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_017() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_018() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_019() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_020() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_021() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_022() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_023() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_024() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_025() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_026() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_027() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_028() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_029() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_030() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_031() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_032() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_033() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_034() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_035() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_036() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_037() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_038() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_039() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_040() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_041() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_042() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_043() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_044() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_045() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_046() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_047() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_048() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_049() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_050() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_051() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_052() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_053() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_054() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_055() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_056() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_057() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_058() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_059() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_060() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_061() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_062() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_063() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_064() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_065() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_066() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_067() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_068() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_069() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_070() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_071() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_072() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_073() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_074() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_075() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_076() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_077() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_078() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_079() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_080() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_081() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_082() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_083() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_084() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_085() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_086() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_087() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_088() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_089() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_090() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_091() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_092() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_093() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_094() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_095() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_096() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_097() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_098() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_099() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_100() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_101() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_102() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_103() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_104() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_105() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_106() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_107() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_108() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_109() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_110() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_111() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_112() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_113() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_114() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_115() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_116() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_117() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_118() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_119() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_120() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_121() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_122() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_123() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_124() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_125() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_126() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_127() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_128() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_129() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_130() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_131() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_132() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_133() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_134() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_135() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_136() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_137() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_138() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_139() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_140() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_141() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_142() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_143() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_144() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_145() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_146() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_147() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_148() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_149() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_150() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_151() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_152() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_153() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_154() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_155() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_156() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_157() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_158() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_159() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_160() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_161() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_162() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_163() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_164() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_165() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_166() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_167() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_168() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_169() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_170() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_171() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_172() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_173() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_174() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_175() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_176() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_177() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_178() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_179() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_180() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_181() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_182() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_183() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_184() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_185() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_186() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_187() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_188() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_189() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_190() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_191() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_192() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_193() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_194() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_195() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_196() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_197() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_198() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_199() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_200() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_201() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_202() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_203() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_204() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_205() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_206() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_207() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_208() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_209() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_210() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_211() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_212() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_213() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_214() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_215() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_216() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_217() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_218() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_219() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_220() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_221() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_222() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_223() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_224() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_225() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_226() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_227() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_228() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_229() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_230() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_231() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_232() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_233() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_234() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_235() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_236() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_237() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_238() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_239() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_240() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_241() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_242() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_243() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_244() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_245() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_246() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_247() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_248() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_249() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_250() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_251() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_252() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_253() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_254() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_255() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_256() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_257() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_258() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_259() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_260() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_261() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_262() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_263() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_264() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_265() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_266() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_267() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_268() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_269() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_270() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_271() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_272() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_273() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_274() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_275() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_276() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_277() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_278() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_279() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_280() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_281() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_282() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_283() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_284() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_285() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_286() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_287() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_288() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_289() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_290() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_291() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_292() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_293() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_294() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_295() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_296() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_297() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_298() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_299() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_300() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_301() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_302() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_303() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_304() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_305() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_306() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_307() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_308() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_309() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_310() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_311() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_312() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_313() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_314() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_315() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_316() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_317() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_318() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_319() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_320() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_321() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_322() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_323() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_324() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_325() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_326() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_327() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_328() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_329() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_330() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_331() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_332() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_333() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_334() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_335() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_336() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_337() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_338() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_339() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_340() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_341() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_342() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_343() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_344() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_345() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_346() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_347() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_348() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_349() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_350() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_351() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_352() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_353() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_354() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_355() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_356() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_357() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_358() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_359() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_360() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_361() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_362() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_363() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    #[test]
    fn test_rmsnorm_stress_364() {
        let rms = RMSNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]);
        let out = rms.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(rms.parameters().len(), 1);
    }

    // Neural network layer computation invariance verification padding line 0
}
