//! # Group & Instance Normalization Layers
//!
//! Divides channels into groups for normalization (GroupNorm) or normalizes each channel instance independently.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RegError, RegKind, RegResult, Regularization};

/// Configuration for Group Normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupNormConfig {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f64,
    pub affine: bool,
}

impl Default for GroupNormConfig {
    fn default() -> Self {
        Self {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        }
    }
}

/// Group Normalization Layer.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    pub config: GroupNormConfig,
    pub weight: Option<Vec<f64>>,
    pub bias: Option<Vec<f64>>,
}

impl GroupNorm {
    pub fn new(config: GroupNormConfig) -> Self {
        let c = config.num_channels;
        let weight = if config.affine { Some(vec![1.0; c]) } else { None };
        let bias = if config.affine { Some(vec![0.0; c]) } else { None };

        Self {
            config,
            weight,
            bias,
        }
    }
}

impl Regularization for GroupNorm {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(RegError::ShapeMismatch {
                expected: vec![1, self.config.num_channels],
                found: shape.to_vec(),
            });
        }

        let batch_size = shape[0];
        let num_channels = shape[1];
        if num_channels != self.config.num_channels {
            return Err(RegError::ShapeMismatch {
                expected: vec![batch_size, self.config.num_channels],
                found: shape.to_vec(),
            });
        }

        let num_groups = self.config.num_groups;
        if num_channels % num_groups != 0 {
            return Err(RegError::InvalidGroupCount {
                num_groups,
                num_channels,
            });
        }

        let channels_per_group = num_channels / num_groups;
        let spatial_size: usize = shape.iter().skip(2).product::<usize>().max(1);
        let group_elements = (channels_per_group * spatial_size) as f64;

        let data = input.data();
        let mut out_data = vec![0.0; data.len()];
        let eps = self.config.eps;

        for b in 0..batch_size {
            for g in 0..num_groups {
                let mut sum = 0.0;
                for c_idx in 0..channels_per_group {
                    let c = g * channels_per_group + c_idx;
                    let offset = (b * num_channels + c) * spatial_size;
                    for s in 0..spatial_size {
                        sum += data[offset + s];
                    }
                }
                let mean = sum / group_elements;

                let mut sq_diff = 0.0;
                for c_idx in 0..channels_per_group {
                    let c = g * channels_per_group + c_idx;
                    let offset = (b * num_channels + c) * spatial_size;
                    for s in 0..spatial_size {
                        let d = data[offset + s] - mean;
                        sq_diff += d * d;
                    }
                }
                let var = sq_diff / group_elements;
                let std_inv = 1.0 / (var + eps).sqrt();

                for c_idx in 0..channels_per_group {
                    let c = g * channels_per_group + c_idx;
                    let offset = (b * num_channels + c) * spatial_size;
                    let gamma = self.weight.as_ref().map(|w| w[c]).unwrap_or(1.0);
                    let beta = self.bias.as_ref().map(|b| b[c]).unwrap_or(0.0);

                    for s in 0..spatial_size {
                        let normalized = (data[offset + s] - mean) * std_inv;
                        out_data[offset + s] = gamma * normalized + beta;
                    }
                }
            }
        }

        Ok(Tensor::from_slice(&out_data, shape.to_vec()))
    }

    fn kind(&self) -> RegKind {
        RegKind::GroupNorm
    }
}

/// Configuration for Instance Normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceNormConfig {
    pub num_features: usize,
    pub eps: f64,
    pub affine: bool,
}

impl Default for InstanceNormConfig {
    fn default() -> Self {
        Self {
            num_features: 1,
            eps: 1e-5,
            affine: false,
        }
    }
}

/// 1D Instance Normalization Layer.
#[derive(Debug, Clone)]
pub struct InstanceNorm1d {
    pub group_norm: GroupNorm,
}

impl InstanceNorm1d {
    pub fn new(config: InstanceNormConfig) -> Self {
        let gn_cfg = GroupNormConfig {
            num_groups: config.num_features,
            num_channels: config.num_features,
            eps: config.eps,
            affine: config.affine,
        };
        Self {
            group_norm: GroupNorm::new(gn_cfg),
        }
    }
}

impl Regularization for InstanceNorm1d {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        self.group_norm.apply(input)
    }

    fn kind(&self) -> RegKind {
        RegKind::InstanceNorm
    }
}

pub type InstanceNorm2d = InstanceNorm1d;
pub type InstanceNorm3d = InstanceNorm1d;

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
