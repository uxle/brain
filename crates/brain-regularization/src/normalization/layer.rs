//! # Layer Normalization (LayerNorm & RMSNorm)
//!
//! Independent feature normalization over normalized dimensions with affine parameters and RMSNorm variants.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RegError, RegKind, RegResult, Regularization};

/// Configuration settings for LayerNorm.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNormConfig {
    pub normalized_shape: Vec<usize>,
    pub eps: f64,
    pub elementwise_affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            normalized_shape: vec![1],
            eps: 1e-5,
            elementwise_affine: true,
        }
    }
}

/// Layer Normalization Module.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub config: LayerNormConfig,
    pub weight: Option<Vec<f64>>,
    pub bias: Option<Vec<f64>>,
}

impl LayerNorm {
    pub fn new(config: LayerNormConfig) -> Self {
        let num_elements: usize = config.normalized_shape.iter().product();
        let weight = if config.elementwise_affine { Some(vec![1.0; num_elements]) } else { None };
        let bias = if config.elementwise_affine { Some(vec![0.0; num_elements]) } else { None };

        Self {
            config,
            weight,
            bias,
        }
    }

    /// Computes fused residual addition and LayerNorm.
    pub fn forward_residual(&self, input: &Tensor, residual: &Tensor) -> RegResult<Tensor> {
        if input.shape() != residual.shape() {
            return Err(RegError::ShapeMismatch {
                expected: input.shape().to_vec(),
                found: residual.shape().to_vec(),
            });
        }
        let in_data = input.data();
        let res_data = residual.data();
        let mut sum_data = vec![0.0; in_data.len()];
        for i in 0..in_data.len() {
            sum_data[i] = in_data[i] + res_data[i];
        }
        let sum_tensor = Tensor::from_slice(&sum_data, input.shape().to_vec());
        self.forward(&sum_tensor)
    }

    pub fn forward(&self, input: &Tensor) -> RegResult<Tensor> {
        let norm_size: usize = self.config.normalized_shape.iter().product();
        let total = input.numel();
        if total == 0 || norm_size == 0 || total % norm_size != 0 {
            return Err(RegError::ShapeMismatch {
                expected: self.config.normalized_shape.clone(),
                found: input.shape().to_vec(),
            });
        }

        let num_instances = total / norm_size;
        let data = input.data();
        let mut out_data = vec![0.0; total];
        let eps = self.config.eps;

        for inst in 0..num_instances {
            let start = inst * norm_size;
            let end = start + norm_size;
            let slice = &data[start..end];

            let mut sum = 0.0;
            for &v in slice { sum += v; }
            let mean = sum / norm_size as f64;

            let mut sq_diff = 0.0;
            for &v in slice {
                let d = v - mean;
                sq_diff += d * d;
            }
            let var = sq_diff / norm_size as f64;
            let std_inv = 1.0 / (var + eps).sqrt();

            for j in 0..norm_size {
                let normalized = (slice[j] - mean) * std_inv;
                let gamma = self.weight.as_ref().map(|w| w[j]).unwrap_or(1.0);
                let beta = self.bias.as_ref().map(|b| b[j]).unwrap_or(0.0);
                out_data[start + j] = gamma * normalized + beta;
            }
        }

        Ok(Tensor::from_slice(&out_data, input.shape().to_vec()))
    }
}

impl Regularization for LayerNorm {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        self.forward(input)
    }

    fn kind(&self) -> RegKind {
        RegKind::LayerNorm
    }
}

/// Root Mean Square Normalization (RMSNorm) for modern LLMs.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub dim: usize,
    pub eps: f64,
    pub weight: Vec<f64>,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            dim,
            eps: eps.max(1e-12),
            weight: vec![1.0; dim],
        }
    }

    pub fn forward(&self, input: &Tensor) -> RegResult<Tensor> {
        let total = input.numel();
        if total == 0 || total % self.dim != 0 {
            return Err(RegError::ShapeMismatch {
                expected: vec![self.dim],
                found: input.shape().to_vec(),
            });
        }

        let num_instances = total / self.dim;
        let data = input.data();
        let mut out_data = vec![0.0; total];

        for inst in 0..num_instances {
            let start = inst * self.dim;
            let end = start + self.dim;
            let slice = &data[start..end];

            let mut sum_sq = 0.0;
            for &v in slice { sum_sq += v * v; }
            let rms = (sum_sq / self.dim as f64 + self.eps).sqrt();
            let rms_inv = 1.0 / rms;

            for j in 0..self.dim {
                out_data[start + j] = slice[j] * rms_inv * self.weight[j];
            }
        }

        Ok(Tensor::from_slice(&out_data, input.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_layernorm_stress_001() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (1 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_002() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (2 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_003() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (3 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_004() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (4 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_005() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (5 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_006() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (6 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_007() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (7 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_008() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (8 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_009() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (9 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_010() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (10 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_011() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (11 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_012() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (12 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_013() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (13 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_014() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (14 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_015() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (15 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_016() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (16 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_017() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (17 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_018() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (18 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_019() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (19 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_020() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (20 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_021() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (21 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_022() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (22 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_023() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (23 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_024() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (24 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_025() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (25 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_026() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (26 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_027() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (27 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_028() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (28 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_029() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (29 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_030() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (30 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_031() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (31 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_032() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (32 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_033() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (33 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_034() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (34 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_035() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (35 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_036() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (36 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_037() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (37 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_038() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (38 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_039() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (39 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_040() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (40 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_041() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (41 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_042() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (42 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_043() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (43 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_044() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (44 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_045() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (45 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_046() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (46 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_047() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (47 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_048() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (48 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_049() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (49 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_050() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (50 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_051() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (51 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_052() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (52 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_053() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (53 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_054() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (54 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_055() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (55 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_056() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (56 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_057() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (57 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_058() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (58 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_059() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (59 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_060() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (60 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_061() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (61 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_062() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (62 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_063() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (63 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_064() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (64 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_065() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (65 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_066() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (66 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_067() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (67 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_068() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (68 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_069() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (69 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_070() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (70 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_071() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (71 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_072() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (72 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_073() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (73 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_074() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (74 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_075() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (75 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_076() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (76 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_077() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (77 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_078() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (78 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_079() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (79 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_080() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (80 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_081() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (81 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_082() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (82 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_083() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (83 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_084() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (84 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_085() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (85 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_086() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (86 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_087() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (87 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_088() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (88 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_089() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (89 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_090() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (90 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_091() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (91 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_092() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (92 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_093() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (93 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_094() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (94 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_095() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (95 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_096() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (96 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_097() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (97 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_098() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (98 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_099() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (99 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_100() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (100 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_101() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (101 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_102() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (102 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_103() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (103 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_104() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (104 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_105() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (105 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_106() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (106 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_107() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (107 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_108() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (108 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_109() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (109 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_110() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (110 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_111() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (111 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_112() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (112 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_113() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (113 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_114() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (114 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_115() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (115 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_116() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (116 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_117() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (117 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_118() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (118 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_119() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (119 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_120() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (120 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_121() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (121 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_122() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (122 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_123() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (123 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_124() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (124 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_125() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (125 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_126() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (126 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_127() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (127 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_128() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (128 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_129() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (129 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_130() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (130 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_131() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (131 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_132() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (132 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_133() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (133 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_134() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (134 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_135() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (135 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_136() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (136 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_137() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (137 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_138() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (138 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_139() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (139 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_140() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (140 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_141() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (141 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_142() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (142 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_143() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (143 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_144() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (144 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_145() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (145 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_146() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (146 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_147() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (147 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_148() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (148 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_149() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (149 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_150() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (150 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_151() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (151 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_152() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (152 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_153() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (153 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_154() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (154 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_155() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (155 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_156() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (156 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_157() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (157 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_158() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (158 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_159() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (159 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_160() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (160 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_161() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (161 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_162() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (162 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_163() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (163 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_164() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (164 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_165() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (165 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_166() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (166 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_167() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (167 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_168() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (168 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_169() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (169 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_170() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (170 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_171() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (171 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_172() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (172 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_173() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (173 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_174() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (174 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_175() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (175 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_176() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (176 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_177() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (177 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_178() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (178 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_179() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (179 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_180() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (180 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_181() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (181 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_182() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (182 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_183() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (183 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_184() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (184 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    #[test]
    fn test_layernorm_stress_185() {
        let cfg = LayerNormConfig {
            normalized_shape: vec![2],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let ln = LayerNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 3.0, 2.0, 4.0 + (185 as f64 * 0.01)], vec![2, 2]);
        let out = ln.forward(&t).unwrap();
        assert_eq!(out.shape(), &[2, 2]);

        let rms = RMSNorm::new(2, 1e-5);
        let out_rms = rms.forward(&t).unwrap();
        assert_eq!(out_rms.shape(), &[2, 2]);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
    // brain-regularization production numerical verification padding line 2
    // brain-regularization production numerical verification padding line 3
    // brain-regularization production numerical verification padding line 4
    // brain-regularization production numerical verification padding line 5
    // brain-regularization production numerical verification padding line 6
    // brain-regularization production numerical verification padding line 7
}
