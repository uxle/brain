//! # Layer Normalization (LayerNorm & RMSNorm)
//!
//! Independent feature normalization over normalized dimensions with affine parameters and RMSNorm variants.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

use super::super::core::{RegError, RegKind, RegResult, Regularization};
use brain_core::Tensor;

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
        let weight = if config.elementwise_affine {
            Some(vec![1.0; num_elements])
        } else {
            None
        };
        let bias = if config.elementwise_affine {
            Some(vec![0.0; num_elements])
        } else {
            None
        };

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
            for &v in slice {
                sum += v;
            }
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
            for &v in slice {
                sum_sq += v * v;
            }
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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown
    )]
    use super::*;
    use crate::augment::*;
    use crate::config::*;
    use crate::consistency::*;
    use crate::core::*;
    use crate::curriculum::*;
    use crate::decay::*;
    use crate::dropout::*;
    use crate::dropout_uncertainty::*;
    use crate::earlystop::*;
    use crate::label_smooth::*;
    use crate::normalization::*;
    use crate::ops::*;
    use crate::perturb::*;
    use crate::r#impl::*;
    use crate::registry::*;
    use crate::regularizers::*;
    use crate::rules::*;
    use crate::stopping::*;
    use crate::train_hooks::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
