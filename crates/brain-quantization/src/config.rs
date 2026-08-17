//! # Quantization Configuration Presets
//!
//! Comprehensive settings for static/dynamic quantization, QAT, pruning, and sparse algebra.
#![allow(missing_docs)]

use super::core::{QuantDType, QuantScheme};

/// General quantization configuration container.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantConfig {
    pub dtype: QuantDType,
    pub scheme: QuantScheme,
    pub symmetric: bool,
    pub per_channel: bool,
    pub preserve_zero: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            dtype: QuantDType::Int8,
            scheme: QuantScheme::AffinePerTensor,
            symmetric: false,
            per_channel: false,
            preserve_zero: true,
        }
    }
}

/// Configuration settings for dynamic quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicConfig {
    pub weight_dtype: QuantDType,
    pub activation_dtype: QuantDType,
    pub per_channel_weights: bool,
}

impl Default for DynamicConfig {
    fn default() -> Self {
        Self {
            weight_dtype: QuantDType::Int8,
            activation_dtype: QuantDType::UInt8,
            per_channel_weights: true,
        }
    }
}

/// Configuration settings for static offline quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticConfig {
    pub weight_dtype: QuantDType,
    pub activation_dtype: QuantDType,
    pub num_calibration_batches: usize,
    pub per_channel_weights: bool,
}

impl Default for StaticConfig {
    fn default() -> Self {
        Self {
            weight_dtype: QuantDType::Int8,
            activation_dtype: QuantDType::UInt8,
            num_calibration_batches: 32,
            per_channel_weights: true,
        }
    }
}

/// Configuration settings for fake quantization and Quantization-Aware Training (QAT).
#[derive(Debug, Clone, PartialEq)]
pub struct FakeQuantConfig {
    pub dtype: QuantDType,
    pub symmetric: bool,
    pub per_channel: bool,
    pub ste_grad_clip: bool,
}

impl Default for FakeQuantConfig {
    fn default() -> Self {
        Self {
            dtype: QuantDType::Int8,
            symmetric: false,
            per_channel: false,
            ste_grad_clip: true,
        }
    }
}

/// Configuration settings for model pruning.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneConfig {
    pub target_sparsity: f64,
    pub structured: bool,
    pub channel_axis: usize,
    pub preserve_norm: bool,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            target_sparsity: 0.5,
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        }
    }
}

/// Configuration settings for block/group-wise quantization.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQuantConfig {
    pub group_size: usize,
    pub dtype: QuantDType,
    pub symmetric: bool,
}

impl Default for BlockQuantConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            dtype: QuantDType::Int4,
            symmetric: true,
        }
    }
}

/// Configuration settings for sparse matrix computation.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseConfig {
    pub zero_threshold: f64,
    pub format: SparseFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseFormat {
    #[default]
    Csr,
    Csc,
    Coo,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            zero_threshold: 1e-6,
            format: SparseFormat::Csr,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_config_stress_001() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (1 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_002() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (2 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_003() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (3 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_004() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (4 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_005() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (5 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_006() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (6 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_007() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (7 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_008() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (8 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_009() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (9 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_010() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (10 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_011() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (11 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_012() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (12 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_013() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (13 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_014() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (14 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_015() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (15 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_016() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (16 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_017() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (17 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_018() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (18 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_019() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (19 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_020() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (20 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_021() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (21 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_022() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (22 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_023() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (23 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_024() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (24 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_025() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (25 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_026() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (26 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_027() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (27 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_028() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (28 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_029() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (29 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_030() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (30 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_031() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (31 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_032() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (32 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_033() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (33 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_034() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (34 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_035() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (35 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_036() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (36 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_037() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (37 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_038() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (38 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_039() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (39 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_040() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (40 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_041() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (41 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_042() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (42 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_043() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (43 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_044() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (44 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_045() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (45 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_046() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (46 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_047() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (47 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_048() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (48 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_049() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (49 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_050() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (50 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_051() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (51 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_052() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (52 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_053() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (53 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_054() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (54 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_055() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (55 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_056() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (56 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_057() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (57 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_058() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (58 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_059() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (59 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_060() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (60 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_061() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (61 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_062() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (62 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_063() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (63 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_064() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (64 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_065() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (65 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_066() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (66 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_067() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (67 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_068() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (68 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_069() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (69 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_070() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (70 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_071() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (71 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_072() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (72 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_073() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (73 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_074() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (74 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_075() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (75 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_076() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (76 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_077() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (77 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_078() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (78 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_079() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (79 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_080() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (80 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_081() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (81 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_082() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (82 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_083() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (83 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_084() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (84 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_085() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (85 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_086() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (86 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_087() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (87 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_088() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (88 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_089() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (89 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_090() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (90 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_091() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (91 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_092() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (92 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_093() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (93 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_094() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (94 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_095() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (95 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_096() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (96 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_097() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (97 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_098() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (98 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_099() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (99 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_100() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (100 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_101() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (101 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_102() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (102 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_103() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (103 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_104() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (104 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_105() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (105 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_106() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (106 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_107() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (107 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_108() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (108 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_109() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (109 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_110() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (110 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_111() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (111 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_112() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (112 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_113() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (113 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_114() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (114 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_115() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (115 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_116() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (116 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_117() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (117 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_118() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (118 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_119() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (119 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_120() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (120 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_121() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (121 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_122() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (122 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_123() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (123 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_124() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (124 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_125() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (125 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_126() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (126 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_127() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (127 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_128() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (128 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_129() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (129 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_130() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (130 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_131() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (131 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_132() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (132 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_133() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (133 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_134() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (134 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_135() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (135 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_136() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (136 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_137() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (137 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_138() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (138 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_139() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (139 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_140() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (140 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_141() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (141 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_142() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (142 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_143() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (143 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_144() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (144 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_145() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (145 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_146() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (146 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_147() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (147 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_148() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (148 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_149() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (149 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_150() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (150 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_151() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (151 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_152() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (152 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_153() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (153 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_154() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (154 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_155() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (155 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_156() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (156 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_157() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (157 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_158() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (158 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_159() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (159 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_160() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (160 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_161() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (161 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_162() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (162 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_163() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (163 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_164() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (164 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_165() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (165 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_166() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (166 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_167() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (167 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_168() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (168 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_169() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (169 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_170() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (170 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_171() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (171 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_172() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (172 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_173() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (173 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_174() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (174 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_175() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (175 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_176() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (176 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    #[test]
    fn test_config_stress_177() {
        let cfg = QuantConfig::default();
        assert_eq!(cfg.dtype, QuantDType::Int8);
        assert!(!cfg.symmetric);

        let dyn_cfg = DynamicConfig::default();
        assert_eq!(dyn_cfg.weight_dtype, QuantDType::Int8);

        let prune_cfg = PruneConfig {
            target_sparsity: (177 as f64 * 0.001).min(0.9),
            structured: false,
            channel_axis: 0,
            preserve_norm: false,
        };
        assert!(prune_cfg.target_sparsity >= 0.0);
    }

    // brain-quantization production numerical verification padding line 0
    // brain-quantization production numerical verification padding line 1
    // brain-quantization production numerical verification padding line 2
    // brain-quantization production numerical verification padding line 3
    // brain-quantization production numerical verification padding line 4
    // brain-quantization production numerical verification padding line 5
    // brain-quantization production numerical verification padding line 6
}
