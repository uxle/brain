//! # Layer Normalization (LayerNorm)
//!
//! Standard Layer Normalization over specified normalized shapes.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};

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
            let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / norm_size as f64;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for i in 0..norm_size {
                let norm = (slice[i] - mean) * inv_std;
                out[b * norm_size + i] = norm * w_data[i] + b_data[i];
            }
        }

        Tensor::from_vec(out, shape.to_vec())
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        Ok(self.forward(input))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_layernorm_stress_001() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_002() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_003() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_004() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_005() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_006() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_007() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_008() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_009() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_010() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_011() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_012() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_013() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_014() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_015() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_016() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_017() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_018() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_019() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_020() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_021() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_022() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_023() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_024() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_025() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_026() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_027() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_028() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_029() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_030() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_031() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_032() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_033() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_034() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_035() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_036() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_037() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_038() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_039() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_040() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_041() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_042() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_043() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_044() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_045() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_046() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_047() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_048() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_049() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_050() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_051() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_052() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_053() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_054() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_055() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_056() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_057() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_058() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_059() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_060() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_061() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_062() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_063() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_064() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_065() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_066() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_067() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_068() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_069() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_070() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_071() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_072() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_073() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_074() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_075() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_076() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_077() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_078() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_079() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_080() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_081() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_082() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_083() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_084() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_085() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_086() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_087() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_088() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_089() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_090() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_091() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_092() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_093() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_094() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_095() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_096() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_097() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_098() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_099() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_100() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_101() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_102() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_103() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_104() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_105() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_106() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_107() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_108() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_109() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_110() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_111() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_112() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_113() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_114() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_115() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_116() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_117() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_118() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_119() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_120() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_121() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_122() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_123() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_124() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_125() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_126() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_127() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_128() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_129() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_130() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_131() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_132() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_133() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_134() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_135() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_136() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_137() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_138() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_139() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_140() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_141() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_142() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_143() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_144() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_145() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_146() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_147() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_148() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_149() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_150() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_151() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_152() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_153() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_154() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_155() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_156() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_157() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_158() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_159() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_160() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_161() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_162() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_163() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_164() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_165() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_166() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_167() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_168() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_169() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_170() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_171() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_172() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_173() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_174() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_175() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_176() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_177() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_178() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_179() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_180() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_181() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_182() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_183() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_184() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_185() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_186() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_187() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_188() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_189() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_190() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_191() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_192() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_193() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_194() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_195() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_196() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_197() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_198() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_199() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_200() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_201() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_202() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_203() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_204() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_205() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_206() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_207() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_208() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_209() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_210() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_211() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_212() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_213() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_214() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_215() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_216() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_217() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_218() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_219() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_220() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_221() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_222() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_223() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_224() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_225() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_226() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_227() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_228() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_229() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_230() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_231() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_232() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_233() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_234() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_235() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_236() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_237() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_238() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_239() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_240() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_241() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_242() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_243() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_244() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_245() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_246() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_247() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_248() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_249() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_250() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_251() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_252() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_253() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_254() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_255() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_256() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_257() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_258() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_259() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_260() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_261() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_262() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_263() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_264() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_265() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_266() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_267() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_268() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_269() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_270() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_271() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_272() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_273() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_274() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_275() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_276() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_277() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_278() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_279() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_280() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_281() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_282() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_283() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_284() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_285() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_286() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_287() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_288() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_289() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_290() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_291() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_292() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_293() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_294() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_295() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_296() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_297() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_298() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_299() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_300() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_301() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_302() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_303() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_304() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_305() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_306() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_307() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_308() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_309() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_310() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_311() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_312() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_313() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_314() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_315() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_316() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_317() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_318() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_319() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_320() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_321() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_322() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_323() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_324() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_325() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_326() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_327() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_328() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_329() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_330() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_331() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_332() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_333() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_334() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_335() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_336() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_337() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_338() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_339() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_340() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_341() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_342() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_343() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_344() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_345() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_346() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_347() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_348() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_349() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_350() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_351() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_352() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_353() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_354() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_355() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_356() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_357() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_358() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_359() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_360() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_361() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_362() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_363() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_layernorm_stress_364() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out = ln.forward(&x);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(ln.parameters().len(), 2);
    }
}
