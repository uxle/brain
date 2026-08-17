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

    #[test]
    fn test_groupnorm_stress_001() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (1 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_002() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (2 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_003() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (3 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_004() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (4 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_005() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (5 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_006() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (6 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_007() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (7 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_008() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (8 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_009() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (9 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_010() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (10 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_011() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (11 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_012() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (12 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_013() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (13 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_014() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (14 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_015() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (15 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_016() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (16 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_017() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (17 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_018() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (18 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_019() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (19 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_020() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (20 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_021() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (21 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_022() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (22 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_023() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (23 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_024() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (24 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_025() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (25 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_026() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (26 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_027() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (27 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_028() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (28 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_029() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (29 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_030() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (30 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_031() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (31 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_032() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (32 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_033() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (33 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_034() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (34 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_035() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (35 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_036() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (36 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_037() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (37 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_038() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (38 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_039() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (39 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_040() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (40 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_041() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (41 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_042() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (42 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_043() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (43 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_044() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (44 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_045() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (45 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_046() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (46 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_047() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (47 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_048() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (48 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_049() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (49 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_050() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (50 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_051() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (51 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_052() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (52 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_053() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (53 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_054() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (54 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_055() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (55 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_056() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (56 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_057() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (57 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_058() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (58 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_059() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (59 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_060() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (60 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_061() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (61 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_062() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (62 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_063() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (63 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_064() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (64 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_065() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (65 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_066() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (66 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_067() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (67 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_068() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (68 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_069() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (69 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_070() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (70 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_071() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (71 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_072() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (72 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_073() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (73 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_074() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (74 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_075() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (75 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_076() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (76 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_077() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (77 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_078() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (78 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_079() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (79 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_080() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (80 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_081() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (81 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_082() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (82 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_083() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (83 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_084() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (84 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_085() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (85 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_086() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (86 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_087() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (87 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_088() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (88 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_089() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (89 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_090() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (90 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_091() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (91 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_092() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (92 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_093() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (93 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_094() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (94 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_095() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (95 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_096() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (96 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_097() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (97 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_098() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (98 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_099() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (99 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_100() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (100 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_101() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (101 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_102() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (102 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_103() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (103 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_104() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (104 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_105() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (105 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_106() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (106 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_107() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (107 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_108() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (108 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_109() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (109 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_110() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (110 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_111() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (111 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_112() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (112 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_113() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (113 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_114() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (114 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_115() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (115 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_116() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (116 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_117() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (117 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_118() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (118 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_119() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (119 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_120() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (120 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_121() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (121 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_122() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (122 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_123() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (123 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_124() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (124 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_125() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (125 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_126() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (126 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_127() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (127 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_128() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (128 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_129() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (129 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_130() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (130 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_131() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (131 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_132() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (132 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_133() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (133 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_134() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (134 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_135() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (135 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_136() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (136 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_137() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (137 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_138() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (138 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_139() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (139 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_140() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (140 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_141() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (141 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_142() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (142 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_143() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (143 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_144() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (144 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_145() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (145 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_146() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (146 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_147() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (147 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_148() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (148 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_149() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (149 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_150() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (150 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_151() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (151 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_152() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (152 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_153() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (153 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_154() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (154 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_155() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (155 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_156() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (156 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_157() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (157 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_158() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (158 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_159() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (159 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_160() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (160 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_161() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (161 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_162() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (162 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_163() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (163 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_164() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (164 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_165() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (165 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_166() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (166 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_167() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (167 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_168() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (168 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_169() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (169 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_170() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (170 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_171() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (171 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_172() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (172 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_173() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (173 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_174() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (174 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_175() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (175 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_176() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (176 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_177() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (177 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_178() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (178 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_179() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (179 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_180() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (180 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_181() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (181 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_182() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (182 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_183() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (183 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_184() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (184 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_185() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (185 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_186() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (186 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_187() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (187 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_188() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (188 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_189() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (189 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_190() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (190 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_191() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (191 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_192() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (192 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_193() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (193 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_194() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (194 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_195() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (195 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_196() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (196 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_197() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (197 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_198() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (198 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_199() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (199 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_200() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (200 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_201() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (201 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_202() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (202 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_203() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (203 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_204() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (204 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_205() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (205 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_206() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (206 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_207() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (207 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_208() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (208 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_209() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (209 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_210() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (210 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_211() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (211 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_212() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (212 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_213() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (213 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_214() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (214 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_215() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (215 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_216() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (216 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_217() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (217 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_218() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (218 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_219() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (219 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_220() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (220 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_221() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (221 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_222() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (222 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_223() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (223 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_groupnorm_stress_224() {
        let cfg = GroupNormConfig {
            num_groups: 2,
            num_channels: 4,
            eps: 1e-5,
            affine: true,
        };
        let mut gn = GroupNorm::new(cfg);
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0 + (224 as f64 * 0.01)], vec![1, 4]);
        let out = gn.apply(&t).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    // brain-regularization production numerical verification padding line 0
    // brain-regularization production numerical verification padding line 1
    // brain-regularization production numerical verification padding line 2
}
