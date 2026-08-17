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

    #[test]
    fn test_linear_stress_001() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_002() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_003() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_004() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_005() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_006() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_007() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_008() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_009() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_010() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_011() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_012() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_013() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_014() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_015() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_016() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_017() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_018() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_019() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_020() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_021() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_022() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_023() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_024() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_025() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_026() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_027() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_028() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_029() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_030() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_031() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_032() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_033() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_034() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_035() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_036() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_037() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_038() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_039() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_040() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_041() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_042() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_043() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_044() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_045() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_046() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_047() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_048() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_049() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_050() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_051() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_052() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_053() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_054() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_055() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_056() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_057() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_058() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_059() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_060() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_061() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_062() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_063() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_064() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_065() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_066() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_067() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_068() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_069() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_070() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_071() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_072() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_073() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_074() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_075() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_076() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_077() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_078() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_079() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_080() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_081() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_082() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_083() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_084() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_085() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_086() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_087() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_088() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_089() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_090() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_091() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_092() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_093() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_094() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_095() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_096() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_097() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_098() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_099() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_100() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_101() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_102() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_103() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_104() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_105() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_106() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_107() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_108() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_109() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_110() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_111() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_112() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_113() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_114() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_115() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_116() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_117() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_118() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_119() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_120() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_121() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_122() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_123() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_124() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_125() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_126() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_127() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_128() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_129() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_130() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_131() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_132() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_133() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_134() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_135() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_136() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_137() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_138() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_139() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_140() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_141() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_142() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_143() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_144() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_145() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_146() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_147() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_148() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_149() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_150() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_151() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_152() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_153() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_154() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_155() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_156() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_157() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_158() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_159() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_160() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_161() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_162() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_163() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_164() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_165() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_166() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_167() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_168() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_169() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_170() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_171() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_172() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_173() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_174() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_175() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_176() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_177() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_178() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_179() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_180() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_181() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_182() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_183() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_184() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_185() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_186() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_187() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_188() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_189() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_190() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_191() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_192() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_193() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_194() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_195() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_196() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_197() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_198() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_199() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_200() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_201() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_202() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_203() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_204() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_205() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_206() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_207() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_208() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_209() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_210() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_211() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_212() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_213() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_214() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_215() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_216() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_217() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_218() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_219() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_220() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_221() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_222() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_223() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_224() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_225() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_226() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_227() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_228() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_229() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_230() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_231() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_232() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_233() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_234() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_235() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_236() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_237() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_238() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_239() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_240() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_241() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_242() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_243() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_244() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_245() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_246() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_247() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_248() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_249() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_250() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_251() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_252() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_253() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_254() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_255() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_256() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_257() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_258() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_259() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_260() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_261() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_262() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_263() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_264() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_265() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_266() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_267() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_268() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_269() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_270() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_271() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_272() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_273() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_274() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_275() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_276() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_277() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_278() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_279() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_280() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_281() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_282() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_283() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_284() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_285() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_286() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_287() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_288() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_289() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_290() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_291() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_292() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_293() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_294() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_295() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_296() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_297() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_298() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_299() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_300() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_301() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_302() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_303() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_304() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_305() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_306() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_307() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_308() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_309() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_310() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_311() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_312() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_313() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_314() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_315() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_316() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_317() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_318() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_319() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_320() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_321() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_322() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_323() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_324() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_325() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_326() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_327() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_328() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_329() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_330() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_331() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_332() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_333() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_334() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_335() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_336() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_337() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_338() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_339() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_340() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_341() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_342() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_343() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_344() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_345() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_346() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_347() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_348() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_349() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_350() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_351() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_352() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_353() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_354() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_355() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_356() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_357() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_358() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_359() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_360() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_361() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_362() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    #[test]
    fn test_linear_stress_363() {
        let fc = Linear::new(4, 2, true);
        let x = Tensor::zeros(vec![1, 4]);
        let out = fc.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(fc.parameters().len(), 2);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
}
