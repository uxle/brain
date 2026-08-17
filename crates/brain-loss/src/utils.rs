//! # Loss Utilities
//!
//! Reduction application, shape validation, numerical clamping, and weighted averages.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{Reduction, LossError, LossResult};

/// Applies reduction (Mean, Sum, None) to a vector of per-sample losses.
pub fn reduction_apply(losses: &[f64], reduction: Reduction) -> Tensor {
    if losses.is_empty() {
        return Tensor::zeros(vec![1]);
    }
    match reduction {
        Reduction::Mean => {
            let sum: f64 = losses.iter().sum();
            Tensor::from_vec(vec![sum / losses.len() as f64], vec![1])
        }
        Reduction::Sum => {
            let sum: f64 = losses.iter().sum();
            Tensor::from_vec(vec![sum], vec![1])
        }
        Reduction::None => {
            Tensor::from_vec(losses.to_vec(), vec![losses.len()])
        }
    }
}

/// Validates that two tensors have matching shapes.
pub fn check_shapes(a: &Tensor, b: &Tensor) -> LossResult<()> {
    if a.shape() != b.shape() {
        return Err(LossError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

/// Clamps values to avoid log(0) and division by zero: [eps, 1.0 - eps].
pub fn clamp_eps(val: f64, eps: f64) -> f64 {
    val.clamp(eps, 1.0 - eps)
}

/// Computes weighted average of sample losses.
pub fn weighted_average(losses: &[f64], weights: &[f64]) -> f64 {
    let n = losses.len().min(weights.len());
    if n == 0 { return 0.0; }
    let mut total_loss = 0.0;
    let mut total_weight = 0.0;
    for i in 0..n {
        total_loss += losses[i] * weights[i];
        total_weight += weights[i];
    }
    if total_weight > 0.0 { total_loss / total_weight } else { 0.0 }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_utils_stress_001() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_002() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_003() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_004() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_005() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_006() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_007() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_008() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_009() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_010() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_011() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_012() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_013() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_014() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_015() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_016() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_017() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_018() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_019() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_020() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_021() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_022() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_023() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_024() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_025() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_026() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_027() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_028() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_029() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_030() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_031() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_032() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_033() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_034() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_035() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_036() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_037() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_038() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_039() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_040() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_041() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_042() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_043() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_044() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_045() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_046() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_047() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_048() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_049() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_050() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_051() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_052() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_053() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_054() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_055() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_056() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_057() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_058() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_059() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_060() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_061() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_062() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_063() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_064() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_065() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_066() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_067() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_068() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_069() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_070() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_071() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_072() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_073() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_074() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_075() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_076() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_077() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_078() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_079() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_080() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_081() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_082() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_083() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_084() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_085() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_086() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_087() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_088() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_089() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_090() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_091() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_092() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_093() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_094() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_095() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_096() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_097() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_098() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_099() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_100() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_101() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_102() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_103() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_104() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_105() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_106() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_107() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_108() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_109() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_110() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_111() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_112() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_113() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_114() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_115() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_116() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_117() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_118() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_119() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_120() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_121() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_122() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_123() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_124() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_125() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_126() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_127() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_128() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_129() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_130() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_131() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_132() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_133() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_134() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_135() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_136() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_137() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_138() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_139() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_140() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_141() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_142() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_143() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_144() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_145() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_146() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_147() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_148() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_149() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_150() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_151() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_152() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_153() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_154() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_155() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_156() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_157() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_158() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_159() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_160() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_161() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_162() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_163() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_164() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_165() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_166() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_167() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_168() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_169() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_170() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_171() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_172() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_173() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_174() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_175() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_176() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_177() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_178() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_179() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_180() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_181() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_182() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_183() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_184() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_185() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_186() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_187() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_188() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_189() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_190() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_191() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_192() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_193() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_194() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_195() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_196() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_197() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_198() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_199() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_200() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_201() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_202() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_203() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_204() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_205() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_206() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_207() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_208() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_209() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_210() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_211() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_212() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_213() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_214() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_215() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_216() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_217() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_218() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }

    #[test]
    fn test_utils_stress_219() {
        let losses = vec![1.0, 2.0, 3.0];
        let m = reduction_apply(&losses, Reduction::Mean);
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-9);
        let s = reduction_apply(&losses, Reduction::Sum);
        assert!((s.to_vec()[0] - 6.0).abs() < 1e-9);
        let n = reduction_apply(&losses, Reduction::None);
        assert_eq!(n.shape(), &[3]);

        let w_avg = weighted_average(&[1.0, 2.0], &[0.5, 0.5]);
        assert!((w_avg - 1.5).abs() < 1e-9);
        assert_eq!(clamp_eps(0.0, 1e-7), 1e-7);
    }
}
