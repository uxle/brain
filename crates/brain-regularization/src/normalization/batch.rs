//! # Batch Normalization Layers (BatchNorm1d / BatchNorm2d / BatchNorm3d)
//!
//! Tracks running mean and variance across batches with exponential momentum and affine scale/shift.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RegError, RegKind, RegResult, Regularization};

/// Configuration for Batch Normalization layers.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchNormConfig {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
    pub affine: bool,
    pub track_running_stats: bool,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            num_features: 1,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        }
    }
}

/// 1D Batch Normalization Layer (Operating on `[B, C, L]` or `[B, C]`).
#[derive(Debug, Clone)]
pub struct BatchNorm1d {
    pub config: BatchNormConfig,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub weight: Option<Vec<f64>>,
    pub bias: Option<Vec<f64>>,
    pub is_training: bool,
    pub num_batches_tracked: usize,
}

impl BatchNorm1d {
    pub fn new(config: BatchNormConfig) -> Self {
        let c = config.num_features;
        let weight = if config.affine { Some(vec![1.0; c]) } else { None };
        let bias = if config.affine { Some(vec![0.0; c]) } else { None };

        Self {
            config,
            running_mean: vec![0.0; c],
            running_var: vec![1.0; c],
            weight,
            bias,
            is_training: true,
            num_batches_tracked: 0,
        }
    }
}

impl Regularization for BatchNorm1d {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        let shape = input.shape();
        if shape.is_empty() || shape.len() < 2 {
            return Err(RegError::ShapeMismatch {
                expected: vec![1, self.config.num_features],
                found: shape.to_vec(),
            });
        }

        let batch_size = shape[0];
        let num_channels = shape[1];
        if num_channels != self.config.num_features {
            return Err(RegError::ShapeMismatch {
                expected: vec![batch_size, self.config.num_features],
                found: shape.to_vec(),
            });
        }

        let spatial_size: usize = shape.iter().skip(2).product::<usize>().max(1);
        let m = (batch_size * spatial_size) as f64;
        let data = input.data();
        let mut out_data = vec![0.0; data.len()];

        let eps = self.config.eps;
        let momentum = self.config.momentum;

        for c in 0..num_channels {
            let (mean, var) = if self.is_training {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    let offset = (b * num_channels + c) * spatial_size;
                    for s in 0..spatial_size {
                        sum += data[offset + s];
                    }
                }
                let current_mean = sum / m;

                let mut sq_diff = 0.0;
                for b in 0..batch_size {
                    let offset = (b * num_channels + c) * spatial_size;
                    for s in 0..spatial_size {
                        let d = data[offset + s] - current_mean;
                        sq_diff += d * d;
                    }
                }
                let current_var = sq_diff / m;

                if self.config.track_running_stats {
                    self.running_mean[c] = (1.0 - momentum) * self.running_mean[c] + momentum * current_mean;
                    let unbiased_var = if m > 1.0 { sq_diff / (m - 1.0) } else { current_var };
                    self.running_var[c] = (1.0 - momentum) * self.running_var[c] + momentum * unbiased_var;
                }

                (current_mean, current_var)
            } else {
                (self.running_mean[c], self.running_var[c])
            };

            let std_inv = 1.0 / (var + eps).sqrt();
            let gamma = self.weight.as_ref().map(|w| w[c]).unwrap_or(1.0);
            let beta = self.bias.as_ref().map(|b| b[c]).unwrap_or(0.0);

            for b in 0..batch_size {
                let offset = (b * num_channels + c) * spatial_size;
                for s in 0..spatial_size {
                    let normalized = (data[offset + s] - mean) * std_inv;
                    out_data[offset + s] = gamma * normalized + beta;
                }
            }
        }

        if self.is_training && self.config.track_running_stats {
            self.num_batches_tracked += 1;
        }

        Ok(Tensor::from_slice(&out_data, shape.to_vec()))
    }

    fn train_mode(&mut self) {
        self.is_training = true;
    }

    fn eval_mode(&mut self) {
        self.is_training = false;
    }

    fn kind(&self) -> RegKind {
        RegKind::BatchNorm
    }
}

/// 2D Spatial Batch Normalization Layer (Operating on `[B, C, H, W]`).
pub type BatchNorm2d = BatchNorm1d;

/// 3D Volumetric Batch Normalization Layer (Operating on `[B, C, D, H, W]`).
pub type BatchNorm3d = BatchNorm1d;

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
}
