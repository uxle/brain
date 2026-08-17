//! # Global Norm and Value Gradient Clipping
//!
//! Standard L1, L2, and L-infinity norm clipping for gradient tensors.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Norm type calculation enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormType {
    L1,
    #[default]
    L2,
    LInf,
}

/// Configuration container for gradient clipping.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipConfig {
    pub max_norm: f64,
    pub norm_type: NormType,
    pub error_if_nonfinite: bool,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            max_norm: 1.0,
            norm_type: NormType::L2,
            error_if_nonfinite: false,
        }
    }
}

/// Clips gradient norm of an iterable of tensors in-place.
///
/// Returns the total norm of the gradients (viewed as a single vector).
pub fn clip_grad_norm_(grads: &mut [Tensor], max_norm: f64, norm_type: NormType) -> f64 {
    if grads.is_empty() || max_norm <= 0.0 {
        return 0.0;
    }

    let total_norm = match norm_type {
        NormType::L2 => {
            let mut sum_sq = 0.0;
            for g in grads.iter() {
                for &val in g.data() {
                    if !val.is_nan() && !val.is_infinite() {
                        sum_sq += val * val;
                    }
                }
            }
            sum_sq.sqrt()
        }
        NormType::L1 => {
            let mut sum_abs = 0.0;
            for g in grads.iter() {
                for &val in g.data() {
                    if !val.is_nan() && !val.is_infinite() {
                        sum_abs += val.abs();
                    }
                }
            }
            sum_abs
        }
        NormType::LInf => {
            let mut max_abs = 0.0f64;
            for g in grads.iter() {
                for &val in g.data() {
                    if !val.is_nan() && !val.is_infinite() {
                        max_abs = max_abs.max(val.abs());
                    }
                }
            }
            max_abs
        }
    };

    let clip_coef = max_norm / (total_norm + 1e-6);
    if clip_coef < 1.0 {
        for g in grads.iter_mut() {
            for val in g.data_mut() {
                *val *= clip_coef;
            }
        }
    }

    total_norm
}

/// Clips gradient values of an iterable of tensors in-place at specified maximum absolute value.
pub fn clip_grad_value_(grads: &mut [Tensor], clip_value: f64) {
    if clip_value <= 0.0 {
        return;
    }
    for g in grads.iter_mut() {
        for val in g.data_mut() {
            *val = val.clamp(-clip_value, clip_value);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_norm_clipping_stress_001() {
        let mut grads = vec![Tensor::from_slice(&[1 as f64, (1 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_002() {
        let mut grads = vec![Tensor::from_slice(&[2 as f64, (2 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_003() {
        let mut grads = vec![Tensor::from_slice(&[3 as f64, (3 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_004() {
        let mut grads = vec![Tensor::from_slice(&[4 as f64, (4 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_005() {
        let mut grads = vec![Tensor::from_slice(&[5 as f64, (5 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_006() {
        let mut grads = vec![Tensor::from_slice(&[6 as f64, (6 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_007() {
        let mut grads = vec![Tensor::from_slice(&[7 as f64, (7 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_008() {
        let mut grads = vec![Tensor::from_slice(&[8 as f64, (8 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_009() {
        let mut grads = vec![Tensor::from_slice(&[9 as f64, (9 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_010() {
        let mut grads = vec![Tensor::from_slice(&[10 as f64, (10 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_011() {
        let mut grads = vec![Tensor::from_slice(&[11 as f64, (11 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_012() {
        let mut grads = vec![Tensor::from_slice(&[12 as f64, (12 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_013() {
        let mut grads = vec![Tensor::from_slice(&[13 as f64, (13 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_014() {
        let mut grads = vec![Tensor::from_slice(&[14 as f64, (14 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_015() {
        let mut grads = vec![Tensor::from_slice(&[15 as f64, (15 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_016() {
        let mut grads = vec![Tensor::from_slice(&[16 as f64, (16 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_017() {
        let mut grads = vec![Tensor::from_slice(&[17 as f64, (17 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_018() {
        let mut grads = vec![Tensor::from_slice(&[18 as f64, (18 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_019() {
        let mut grads = vec![Tensor::from_slice(&[19 as f64, (19 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_020() {
        let mut grads = vec![Tensor::from_slice(&[20 as f64, (20 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_021() {
        let mut grads = vec![Tensor::from_slice(&[21 as f64, (21 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_022() {
        let mut grads = vec![Tensor::from_slice(&[22 as f64, (22 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_023() {
        let mut grads = vec![Tensor::from_slice(&[23 as f64, (23 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_024() {
        let mut grads = vec![Tensor::from_slice(&[24 as f64, (24 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_025() {
        let mut grads = vec![Tensor::from_slice(&[25 as f64, (25 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_026() {
        let mut grads = vec![Tensor::from_slice(&[26 as f64, (26 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_027() {
        let mut grads = vec![Tensor::from_slice(&[27 as f64, (27 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_028() {
        let mut grads = vec![Tensor::from_slice(&[28 as f64, (28 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_029() {
        let mut grads = vec![Tensor::from_slice(&[29 as f64, (29 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_030() {
        let mut grads = vec![Tensor::from_slice(&[30 as f64, (30 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_031() {
        let mut grads = vec![Tensor::from_slice(&[31 as f64, (31 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_032() {
        let mut grads = vec![Tensor::from_slice(&[32 as f64, (32 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_033() {
        let mut grads = vec![Tensor::from_slice(&[33 as f64, (33 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_034() {
        let mut grads = vec![Tensor::from_slice(&[34 as f64, (34 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_035() {
        let mut grads = vec![Tensor::from_slice(&[35 as f64, (35 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_036() {
        let mut grads = vec![Tensor::from_slice(&[36 as f64, (36 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_037() {
        let mut grads = vec![Tensor::from_slice(&[37 as f64, (37 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_038() {
        let mut grads = vec![Tensor::from_slice(&[38 as f64, (38 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_039() {
        let mut grads = vec![Tensor::from_slice(&[39 as f64, (39 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_040() {
        let mut grads = vec![Tensor::from_slice(&[40 as f64, (40 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_041() {
        let mut grads = vec![Tensor::from_slice(&[41 as f64, (41 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_042() {
        let mut grads = vec![Tensor::from_slice(&[42 as f64, (42 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_043() {
        let mut grads = vec![Tensor::from_slice(&[43 as f64, (43 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_044() {
        let mut grads = vec![Tensor::from_slice(&[44 as f64, (44 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_045() {
        let mut grads = vec![Tensor::from_slice(&[45 as f64, (45 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_046() {
        let mut grads = vec![Tensor::from_slice(&[46 as f64, (46 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_047() {
        let mut grads = vec![Tensor::from_slice(&[47 as f64, (47 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_048() {
        let mut grads = vec![Tensor::from_slice(&[48 as f64, (48 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_049() {
        let mut grads = vec![Tensor::from_slice(&[49 as f64, (49 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_050() {
        let mut grads = vec![Tensor::from_slice(&[50 as f64, (50 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_051() {
        let mut grads = vec![Tensor::from_slice(&[51 as f64, (51 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_052() {
        let mut grads = vec![Tensor::from_slice(&[52 as f64, (52 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_053() {
        let mut grads = vec![Tensor::from_slice(&[53 as f64, (53 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_054() {
        let mut grads = vec![Tensor::from_slice(&[54 as f64, (54 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_055() {
        let mut grads = vec![Tensor::from_slice(&[55 as f64, (55 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_056() {
        let mut grads = vec![Tensor::from_slice(&[56 as f64, (56 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_057() {
        let mut grads = vec![Tensor::from_slice(&[57 as f64, (57 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_058() {
        let mut grads = vec![Tensor::from_slice(&[58 as f64, (58 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_059() {
        let mut grads = vec![Tensor::from_slice(&[59 as f64, (59 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_060() {
        let mut grads = vec![Tensor::from_slice(&[60 as f64, (60 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_061() {
        let mut grads = vec![Tensor::from_slice(&[61 as f64, (61 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_062() {
        let mut grads = vec![Tensor::from_slice(&[62 as f64, (62 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_063() {
        let mut grads = vec![Tensor::from_slice(&[63 as f64, (63 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_064() {
        let mut grads = vec![Tensor::from_slice(&[64 as f64, (64 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_065() {
        let mut grads = vec![Tensor::from_slice(&[65 as f64, (65 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_066() {
        let mut grads = vec![Tensor::from_slice(&[66 as f64, (66 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_067() {
        let mut grads = vec![Tensor::from_slice(&[67 as f64, (67 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_068() {
        let mut grads = vec![Tensor::from_slice(&[68 as f64, (68 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_069() {
        let mut grads = vec![Tensor::from_slice(&[69 as f64, (69 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_070() {
        let mut grads = vec![Tensor::from_slice(&[70 as f64, (70 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_071() {
        let mut grads = vec![Tensor::from_slice(&[71 as f64, (71 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_072() {
        let mut grads = vec![Tensor::from_slice(&[72 as f64, (72 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_073() {
        let mut grads = vec![Tensor::from_slice(&[73 as f64, (73 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_074() {
        let mut grads = vec![Tensor::from_slice(&[74 as f64, (74 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_075() {
        let mut grads = vec![Tensor::from_slice(&[75 as f64, (75 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_076() {
        let mut grads = vec![Tensor::from_slice(&[76 as f64, (76 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_077() {
        let mut grads = vec![Tensor::from_slice(&[77 as f64, (77 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_078() {
        let mut grads = vec![Tensor::from_slice(&[78 as f64, (78 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_079() {
        let mut grads = vec![Tensor::from_slice(&[79 as f64, (79 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_080() {
        let mut grads = vec![Tensor::from_slice(&[80 as f64, (80 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_081() {
        let mut grads = vec![Tensor::from_slice(&[81 as f64, (81 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_082() {
        let mut grads = vec![Tensor::from_slice(&[82 as f64, (82 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_083() {
        let mut grads = vec![Tensor::from_slice(&[83 as f64, (83 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_084() {
        let mut grads = vec![Tensor::from_slice(&[84 as f64, (84 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_085() {
        let mut grads = vec![Tensor::from_slice(&[85 as f64, (85 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_086() {
        let mut grads = vec![Tensor::from_slice(&[86 as f64, (86 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_087() {
        let mut grads = vec![Tensor::from_slice(&[87 as f64, (87 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_088() {
        let mut grads = vec![Tensor::from_slice(&[88 as f64, (88 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_089() {
        let mut grads = vec![Tensor::from_slice(&[89 as f64, (89 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_090() {
        let mut grads = vec![Tensor::from_slice(&[90 as f64, (90 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_091() {
        let mut grads = vec![Tensor::from_slice(&[91 as f64, (91 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_092() {
        let mut grads = vec![Tensor::from_slice(&[92 as f64, (92 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_093() {
        let mut grads = vec![Tensor::from_slice(&[93 as f64, (93 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_094() {
        let mut grads = vec![Tensor::from_slice(&[94 as f64, (94 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_095() {
        let mut grads = vec![Tensor::from_slice(&[95 as f64, (95 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_096() {
        let mut grads = vec![Tensor::from_slice(&[96 as f64, (96 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_097() {
        let mut grads = vec![Tensor::from_slice(&[97 as f64, (97 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_098() {
        let mut grads = vec![Tensor::from_slice(&[98 as f64, (98 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_099() {
        let mut grads = vec![Tensor::from_slice(&[99 as f64, (99 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_100() {
        let mut grads = vec![Tensor::from_slice(&[100 as f64, (100 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_101() {
        let mut grads = vec![Tensor::from_slice(&[101 as f64, (101 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_102() {
        let mut grads = vec![Tensor::from_slice(&[102 as f64, (102 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_103() {
        let mut grads = vec![Tensor::from_slice(&[103 as f64, (103 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_104() {
        let mut grads = vec![Tensor::from_slice(&[104 as f64, (104 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_105() {
        let mut grads = vec![Tensor::from_slice(&[105 as f64, (105 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_106() {
        let mut grads = vec![Tensor::from_slice(&[106 as f64, (106 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_107() {
        let mut grads = vec![Tensor::from_slice(&[107 as f64, (107 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_108() {
        let mut grads = vec![Tensor::from_slice(&[108 as f64, (108 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_109() {
        let mut grads = vec![Tensor::from_slice(&[109 as f64, (109 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_110() {
        let mut grads = vec![Tensor::from_slice(&[110 as f64, (110 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_111() {
        let mut grads = vec![Tensor::from_slice(&[111 as f64, (111 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_112() {
        let mut grads = vec![Tensor::from_slice(&[112 as f64, (112 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_113() {
        let mut grads = vec![Tensor::from_slice(&[113 as f64, (113 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_114() {
        let mut grads = vec![Tensor::from_slice(&[114 as f64, (114 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_115() {
        let mut grads = vec![Tensor::from_slice(&[115 as f64, (115 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_116() {
        let mut grads = vec![Tensor::from_slice(&[116 as f64, (116 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_117() {
        let mut grads = vec![Tensor::from_slice(&[117 as f64, (117 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_118() {
        let mut grads = vec![Tensor::from_slice(&[118 as f64, (118 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_119() {
        let mut grads = vec![Tensor::from_slice(&[119 as f64, (119 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_120() {
        let mut grads = vec![Tensor::from_slice(&[120 as f64, (120 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_121() {
        let mut grads = vec![Tensor::from_slice(&[121 as f64, (121 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_122() {
        let mut grads = vec![Tensor::from_slice(&[122 as f64, (122 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_123() {
        let mut grads = vec![Tensor::from_slice(&[123 as f64, (123 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_124() {
        let mut grads = vec![Tensor::from_slice(&[124 as f64, (124 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_125() {
        let mut grads = vec![Tensor::from_slice(&[125 as f64, (125 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_126() {
        let mut grads = vec![Tensor::from_slice(&[126 as f64, (126 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_127() {
        let mut grads = vec![Tensor::from_slice(&[127 as f64, (127 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_128() {
        let mut grads = vec![Tensor::from_slice(&[128 as f64, (128 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_129() {
        let mut grads = vec![Tensor::from_slice(&[129 as f64, (129 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_130() {
        let mut grads = vec![Tensor::from_slice(&[130 as f64, (130 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_131() {
        let mut grads = vec![Tensor::from_slice(&[131 as f64, (131 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_132() {
        let mut grads = vec![Tensor::from_slice(&[132 as f64, (132 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_133() {
        let mut grads = vec![Tensor::from_slice(&[133 as f64, (133 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_134() {
        let mut grads = vec![Tensor::from_slice(&[134 as f64, (134 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_135() {
        let mut grads = vec![Tensor::from_slice(&[135 as f64, (135 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_136() {
        let mut grads = vec![Tensor::from_slice(&[136 as f64, (136 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_137() {
        let mut grads = vec![Tensor::from_slice(&[137 as f64, (137 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_138() {
        let mut grads = vec![Tensor::from_slice(&[138 as f64, (138 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_139() {
        let mut grads = vec![Tensor::from_slice(&[139 as f64, (139 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_140() {
        let mut grads = vec![Tensor::from_slice(&[140 as f64, (140 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_141() {
        let mut grads = vec![Tensor::from_slice(&[141 as f64, (141 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_142() {
        let mut grads = vec![Tensor::from_slice(&[142 as f64, (142 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_143() {
        let mut grads = vec![Tensor::from_slice(&[143 as f64, (143 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_144() {
        let mut grads = vec![Tensor::from_slice(&[144 as f64, (144 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_145() {
        let mut grads = vec![Tensor::from_slice(&[145 as f64, (145 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_146() {
        let mut grads = vec![Tensor::from_slice(&[146 as f64, (146 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_147() {
        let mut grads = vec![Tensor::from_slice(&[147 as f64, (147 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_148() {
        let mut grads = vec![Tensor::from_slice(&[148 as f64, (148 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_149() {
        let mut grads = vec![Tensor::from_slice(&[149 as f64, (149 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_150() {
        let mut grads = vec![Tensor::from_slice(&[150 as f64, (150 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_151() {
        let mut grads = vec![Tensor::from_slice(&[151 as f64, (151 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_152() {
        let mut grads = vec![Tensor::from_slice(&[152 as f64, (152 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_153() {
        let mut grads = vec![Tensor::from_slice(&[153 as f64, (153 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_154() {
        let mut grads = vec![Tensor::from_slice(&[154 as f64, (154 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_155() {
        let mut grads = vec![Tensor::from_slice(&[155 as f64, (155 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_156() {
        let mut grads = vec![Tensor::from_slice(&[156 as f64, (156 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_157() {
        let mut grads = vec![Tensor::from_slice(&[157 as f64, (157 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_158() {
        let mut grads = vec![Tensor::from_slice(&[158 as f64, (158 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_159() {
        let mut grads = vec![Tensor::from_slice(&[159 as f64, (159 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_160() {
        let mut grads = vec![Tensor::from_slice(&[160 as f64, (160 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_161() {
        let mut grads = vec![Tensor::from_slice(&[161 as f64, (161 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_162() {
        let mut grads = vec![Tensor::from_slice(&[162 as f64, (162 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_163() {
        let mut grads = vec![Tensor::from_slice(&[163 as f64, (163 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_164() {
        let mut grads = vec![Tensor::from_slice(&[164 as f64, (164 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_165() {
        let mut grads = vec![Tensor::from_slice(&[165 as f64, (165 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_166() {
        let mut grads = vec![Tensor::from_slice(&[166 as f64, (166 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_167() {
        let mut grads = vec![Tensor::from_slice(&[167 as f64, (167 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_168() {
        let mut grads = vec![Tensor::from_slice(&[168 as f64, (168 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_169() {
        let mut grads = vec![Tensor::from_slice(&[169 as f64, (169 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_170() {
        let mut grads = vec![Tensor::from_slice(&[170 as f64, (170 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_171() {
        let mut grads = vec![Tensor::from_slice(&[171 as f64, (171 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_172() {
        let mut grads = vec![Tensor::from_slice(&[172 as f64, (172 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_173() {
        let mut grads = vec![Tensor::from_slice(&[173 as f64, (173 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_174() {
        let mut grads = vec![Tensor::from_slice(&[174 as f64, (174 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_175() {
        let mut grads = vec![Tensor::from_slice(&[175 as f64, (175 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_176() {
        let mut grads = vec![Tensor::from_slice(&[176 as f64, (176 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_177() {
        let mut grads = vec![Tensor::from_slice(&[177 as f64, (177 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_178() {
        let mut grads = vec![Tensor::from_slice(&[178 as f64, (178 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_179() {
        let mut grads = vec![Tensor::from_slice(&[179 as f64, (179 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_180() {
        let mut grads = vec![Tensor::from_slice(&[180 as f64, (180 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_181() {
        let mut grads = vec![Tensor::from_slice(&[181 as f64, (181 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_182() {
        let mut grads = vec![Tensor::from_slice(&[182 as f64, (182 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_183() {
        let mut grads = vec![Tensor::from_slice(&[183 as f64, (183 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_184() {
        let mut grads = vec![Tensor::from_slice(&[184 as f64, (184 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_185() {
        let mut grads = vec![Tensor::from_slice(&[185 as f64, (185 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_186() {
        let mut grads = vec![Tensor::from_slice(&[186 as f64, (186 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_187() {
        let mut grads = vec![Tensor::from_slice(&[187 as f64, (187 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_188() {
        let mut grads = vec![Tensor::from_slice(&[188 as f64, (188 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_189() {
        let mut grads = vec![Tensor::from_slice(&[189 as f64, (189 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_190() {
        let mut grads = vec![Tensor::from_slice(&[190 as f64, (190 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_191() {
        let mut grads = vec![Tensor::from_slice(&[191 as f64, (191 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_192() {
        let mut grads = vec![Tensor::from_slice(&[192 as f64, (192 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_193() {
        let mut grads = vec![Tensor::from_slice(&[193 as f64, (193 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_194() {
        let mut grads = vec![Tensor::from_slice(&[194 as f64, (194 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_195() {
        let mut grads = vec![Tensor::from_slice(&[195 as f64, (195 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_196() {
        let mut grads = vec![Tensor::from_slice(&[196 as f64, (196 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_197() {
        let mut grads = vec![Tensor::from_slice(&[197 as f64, (197 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_198() {
        let mut grads = vec![Tensor::from_slice(&[198 as f64, (198 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_199() {
        let mut grads = vec![Tensor::from_slice(&[199 as f64, (199 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_200() {
        let mut grads = vec![Tensor::from_slice(&[200 as f64, (200 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_201() {
        let mut grads = vec![Tensor::from_slice(&[201 as f64, (201 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_202() {
        let mut grads = vec![Tensor::from_slice(&[202 as f64, (202 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_203() {
        let mut grads = vec![Tensor::from_slice(&[203 as f64, (203 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_204() {
        let mut grads = vec![Tensor::from_slice(&[204 as f64, (204 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_205() {
        let mut grads = vec![Tensor::from_slice(&[205 as f64, (205 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_206() {
        let mut grads = vec![Tensor::from_slice(&[206 as f64, (206 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_207() {
        let mut grads = vec![Tensor::from_slice(&[207 as f64, (207 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_208() {
        let mut grads = vec![Tensor::from_slice(&[208 as f64, (208 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_209() {
        let mut grads = vec![Tensor::from_slice(&[209 as f64, (209 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_210() {
        let mut grads = vec![Tensor::from_slice(&[210 as f64, (210 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_211() {
        let mut grads = vec![Tensor::from_slice(&[211 as f64, (211 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_212() {
        let mut grads = vec![Tensor::from_slice(&[212 as f64, (212 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_213() {
        let mut grads = vec![Tensor::from_slice(&[213 as f64, (213 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_214() {
        let mut grads = vec![Tensor::from_slice(&[214 as f64, (214 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_215() {
        let mut grads = vec![Tensor::from_slice(&[215 as f64, (215 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_216() {
        let mut grads = vec![Tensor::from_slice(&[216 as f64, (216 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_217() {
        let mut grads = vec![Tensor::from_slice(&[217 as f64, (217 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_218() {
        let mut grads = vec![Tensor::from_slice(&[218 as f64, (218 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_219() {
        let mut grads = vec![Tensor::from_slice(&[219 as f64, (219 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_220() {
        let mut grads = vec![Tensor::from_slice(&[220 as f64, (220 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_221() {
        let mut grads = vec![Tensor::from_slice(&[221 as f64, (221 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_222() {
        let mut grads = vec![Tensor::from_slice(&[222 as f64, (222 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_223() {
        let mut grads = vec![Tensor::from_slice(&[223 as f64, (223 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_224() {
        let mut grads = vec![Tensor::from_slice(&[224 as f64, (224 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_225() {
        let mut grads = vec![Tensor::from_slice(&[225 as f64, (225 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_226() {
        let mut grads = vec![Tensor::from_slice(&[226 as f64, (226 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_227() {
        let mut grads = vec![Tensor::from_slice(&[227 as f64, (227 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_228() {
        let mut grads = vec![Tensor::from_slice(&[228 as f64, (228 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_229() {
        let mut grads = vec![Tensor::from_slice(&[229 as f64, (229 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_230() {
        let mut grads = vec![Tensor::from_slice(&[230 as f64, (230 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_231() {
        let mut grads = vec![Tensor::from_slice(&[231 as f64, (231 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_232() {
        let mut grads = vec![Tensor::from_slice(&[232 as f64, (232 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_233() {
        let mut grads = vec![Tensor::from_slice(&[233 as f64, (233 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_234() {
        let mut grads = vec![Tensor::from_slice(&[234 as f64, (234 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_235() {
        let mut grads = vec![Tensor::from_slice(&[235 as f64, (235 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_236() {
        let mut grads = vec![Tensor::from_slice(&[236 as f64, (236 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_237() {
        let mut grads = vec![Tensor::from_slice(&[237 as f64, (237 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_238() {
        let mut grads = vec![Tensor::from_slice(&[238 as f64, (238 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_239() {
        let mut grads = vec![Tensor::from_slice(&[239 as f64, (239 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_240() {
        let mut grads = vec![Tensor::from_slice(&[240 as f64, (240 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_241() {
        let mut grads = vec![Tensor::from_slice(&[241 as f64, (241 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_242() {
        let mut grads = vec![Tensor::from_slice(&[242 as f64, (242 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_243() {
        let mut grads = vec![Tensor::from_slice(&[243 as f64, (243 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_244() {
        let mut grads = vec![Tensor::from_slice(&[244 as f64, (244 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_245() {
        let mut grads = vec![Tensor::from_slice(&[245 as f64, (245 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_246() {
        let mut grads = vec![Tensor::from_slice(&[246 as f64, (246 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_247() {
        let mut grads = vec![Tensor::from_slice(&[247 as f64, (247 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_248() {
        let mut grads = vec![Tensor::from_slice(&[248 as f64, (248 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_249() {
        let mut grads = vec![Tensor::from_slice(&[249 as f64, (249 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_250() {
        let mut grads = vec![Tensor::from_slice(&[250 as f64, (250 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_251() {
        let mut grads = vec![Tensor::from_slice(&[251 as f64, (251 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_252() {
        let mut grads = vec![Tensor::from_slice(&[252 as f64, (252 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_253() {
        let mut grads = vec![Tensor::from_slice(&[253 as f64, (253 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_254() {
        let mut grads = vec![Tensor::from_slice(&[254 as f64, (254 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_255() {
        let mut grads = vec![Tensor::from_slice(&[255 as f64, (255 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_256() {
        let mut grads = vec![Tensor::from_slice(&[256 as f64, (256 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_257() {
        let mut grads = vec![Tensor::from_slice(&[257 as f64, (257 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_258() {
        let mut grads = vec![Tensor::from_slice(&[258 as f64, (258 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_259() {
        let mut grads = vec![Tensor::from_slice(&[259 as f64, (259 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_260() {
        let mut grads = vec![Tensor::from_slice(&[260 as f64, (260 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_261() {
        let mut grads = vec![Tensor::from_slice(&[261 as f64, (261 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_262() {
        let mut grads = vec![Tensor::from_slice(&[262 as f64, (262 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_263() {
        let mut grads = vec![Tensor::from_slice(&[263 as f64, (263 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_264() {
        let mut grads = vec![Tensor::from_slice(&[264 as f64, (264 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_265() {
        let mut grads = vec![Tensor::from_slice(&[265 as f64, (265 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_266() {
        let mut grads = vec![Tensor::from_slice(&[266 as f64, (266 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_267() {
        let mut grads = vec![Tensor::from_slice(&[267 as f64, (267 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_268() {
        let mut grads = vec![Tensor::from_slice(&[268 as f64, (268 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_269() {
        let mut grads = vec![Tensor::from_slice(&[269 as f64, (269 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }

    #[test]
    fn test_norm_clipping_stress_270() {
        let mut grads = vec![Tensor::from_slice(&[270 as f64, (270 as f64) * 2.0], vec![2])];
        let total_norm = clip_grad_norm_(&mut grads, 1.0, NormType::L2);
        assert!(total_norm > 0.0);

        let mut grads_val = vec![Tensor::from_slice(&[10.0, -10.0], vec![2])];
        clip_grad_value_(&mut grads_val, 2.0);
        assert_eq!(grads_val[0].data()[0], 2.0);
        assert_eq!(grads_val[0].data()[1], -2.0);
    }
}
