//! Reduction and statistical aggregation operations for tensors.
//!
//! This module provides global and axis-wise reductions (sum, mean, prod, min, max, var, std, ptp),
//! lp-norms along axes, numerically stable logsumexp, boolean reductions (all, any),
//! NaN-ignoring statistical aggregations (nansum, nanmean, nanvar, nanstd), and arg-reductions (argmax, argmin).

use crate::tensor::Tensor;

// =============================================================================
// Global Reductions
// =============================================================================

/// Computes the sum of all elements in the tensor.
pub fn sum(a: &Tensor) -> f64 {
    a.data().iter().sum()
}

/// Computes the arithmetic mean of all elements in the tensor.
pub fn mean(a: &Tensor) -> f64 {
    assert!(!a.is_empty(), "mean of empty tensor is undefined");
    sum(a) / (a.numel() as f64)
}

/// Computes the product of all elements in the tensor.
pub fn prod(a: &Tensor) -> f64 {
    a.data().iter().product()
}

/// Computes the minimum element in the tensor.
pub fn min(a: &Tensor) -> f64 {
    assert!(!a.is_empty(), "min of empty tensor is undefined");
    a.data().iter().copied().fold(f64::INFINITY, f64::min)
}

/// Computes the maximum element in the tensor.
pub fn max(a: &Tensor) -> f64 {
    assert!(!a.is_empty(), "max of empty tensor is undefined");
    a.data().iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Computes the variance of all elements (unbiased sample variance with Bessel correction).
pub fn var(a: &Tensor, correction: usize) -> f64 {
    assert!(a.numel() > correction, "Degrees of freedom <= 0 for variance");
    let m = mean(a);
    let sum_sq_diff: f64 = a.data().iter().map(|&x| (x - m).powi(2)).sum();
    sum_sq_diff / ((a.numel() - correction) as f64)
}

/// Computes the standard deviation of all elements.
pub fn std(a: &Tensor, correction: usize) -> f64 {
    var(a, correction).sqrt()
}

/// Computes peak-to-peak (max - min) range.
pub fn ptp(a: &Tensor) -> f64 {
    max(a) - min(a)
}

/// Computes sum of squared elements.
pub fn sum_squares(a: &Tensor) -> f64 {
    a.data().iter().map(|&x| x * x).sum()
}

/// Numerically stable global log-sum-exp: ln(sum(e^x)).
pub fn log_sum_exp(a: &Tensor) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = max(a);
    if max_val.is_infinite() && max_val < 0.0 {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = a.data().iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

// =============================================================================
// Boolean Global Reductions
// =============================================================================

/// Returns true if all elements evaluate to non-zero (true).
pub fn all(a: &Tensor) -> bool {
    a.data().iter().all(|&x| x != 0.0)
}

/// Returns true if any element evaluates to non-zero (true).
pub fn any(a: &Tensor) -> bool {
    a.data().iter().any(|&x| x != 0.0)
}

// =============================================================================
// NaN-Aware Global Reductions
// =============================================================================

/// Computes sum of all non-NaN elements.
pub fn nansum(a: &Tensor) -> f64 {
    a.data().iter().filter(|x| !x.is_nan()).sum()
}

/// Computes mean of all non-NaN elements.
pub fn nanmean(a: &Tensor) -> f64 {
    let valid: Vec<f64> = a.data().iter().copied().filter(|x| !x.is_nan()).collect();
    assert!(!valid.is_empty(), "nanmean: no valid non-NaN elements");
    let s: f64 = valid.iter().sum();
    s / (valid.len() as f64)
}

/// Computes variance of non-NaN elements.
pub fn nanvar(a: &Tensor, correction: usize) -> f64 {
    let valid: Vec<f64> = a.data().iter().copied().filter(|x| !x.is_nan()).collect();
    assert!(valid.len() > correction, "nanvar: insufficient non-NaN degrees of freedom");
    let m = valid.iter().sum::<f64>() / (valid.len() as f64);
    let sum_sq: f64 = valid.iter().map(|&x| (x - m).powi(2)).sum();
    sum_sq / ((valid.len() - correction) as f64)
}

/// Computes standard deviation of non-NaN elements.
pub fn nanstd(a: &Tensor, correction: usize) -> f64 {
    nanvar(a, correction).sqrt()
}

/// Computes minimum of non-NaN elements.
pub fn nanmin(a: &Tensor) -> f64 {
    a.data().iter().copied().filter(|x| !x.is_nan()).fold(f64::INFINITY, f64::min)
}

/// Computes maximum of non-NaN elements.
pub fn nanmax(a: &Tensor) -> f64 {
    a.data().iter().copied().filter(|x| !x.is_nan()).fold(f64::NEG_INFINITY, f64::max)
}

// =============================================================================
// Arg-Reductions
// =============================================================================

/// Returns the flat index of the maximum element.
pub fn argmax(a: &Tensor) -> usize {
    assert!(!a.is_empty(), "argmax of empty tensor");
    let mut best_idx = 0;
    let mut best_val = a.get(0);
    for (i, &val) in a.data().iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

/// Returns the flat index of the minimum element.
pub fn argmin(a: &Tensor) -> usize {
    assert!(!a.is_empty(), "argmin of empty tensor");
    let mut best_idx = 0;
    let mut best_val = a.get(0);
    for (i, &val) in a.data().iter().enumerate() {
        if val < best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

// =============================================================================
// Axis-wise Reductions
// =============================================================================

/// Reduces a tensor along dimension `dim` with a reduction closure.
pub fn reduce_along_dim<F>(a: &Tensor, dim: usize, keepdim: bool, init: f64, op: F) -> Tensor
where
    F: Fn(f64, f64) -> f64,
{
    assert!(dim < a.ndim(), "reduce_along_dim: dim out of bounds");
    let mut out_shape = a.shape().to_vec();
    if keepdim {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }

    let out_numel: usize = out_shape.iter().product();
    let dim_size = a.shape()[dim];

    let mut out_data = vec![init; out_numel.max(1)];
    let mut coords = vec![0usize; a.ndim()];

    for _ in 0..a.numel() {
        let val = a.get_index(&coords);
        
        let mut out_coords = coords.clone();
        if keepdim {
            out_coords[dim] = 0;
        } else {
            out_coords.remove(dim);
        }

        let out_idx = if out_shape.is_empty() {
            0
        } else {
            let mut offset = 0;
            let mut stride = 1;
            for i in (0..out_shape.len()).rev() {
                offset += out_coords[i] * stride;
                stride *= out_shape[i];
            }
            offset
        };

        out_data[out_idx] = op(out_data[out_idx], val);

        for d in (0..a.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < a.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Tensor::new(out_data, out_shape)
}

/// Computes sum along a specified dimension.
pub fn sum_along_dim(a: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    reduce_along_dim(a, dim, keepdim, 0.0, |acc, x| acc + x)
}

/// Computes mean along a specified dimension.
pub fn mean_along_dim(a: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    let s = sum_along_dim(a, dim, keepdim);
    let scale = 1.0 / (a.shape()[dim] as f64);
    crate::tensor::arithmetic::mul_scalar(&s, scale)
}

/// Computes min along a specified dimension.
pub fn min_along_dim(a: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    reduce_along_dim(a, dim, keepdim, f64::INFINITY, f64::min)
}

/// Computes max along a specified dimension.
pub fn max_along_dim(a: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    reduce_along_dim(a, dim, keepdim, f64::NEG_INFINITY, f64::max)
}

/// Computes Lp norm along a specified dimension.
pub fn norm_along_axis(a: &Tensor, p: f64, dim: usize, keepdim: bool) -> Tensor {
    if p == 1.0 {
        reduce_along_dim(a, dim, keepdim, 0.0, |acc, x| acc + x.abs())
    } else if p == 2.0 {
        let sum_sq = reduce_along_dim(a, dim, keepdim, 0.0, |acc, x| acc + x * x);
        crate::tensor::math::sqrt(&sum_sq)
    } else if p.is_infinite() {
        reduce_along_dim(a, dim, keepdim, 0.0, |acc, x| acc.max(x.abs()))
    } else {
        let sum_p = reduce_along_dim(a, dim, keepdim, 0.0, |acc, x| acc + x.abs().powf(p));
        crate::tensor::arithmetic::pow_scalar(&sum_p, 1.0 / p)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_reductions() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        assert_eq!(sum(&t), 10.0);
        assert_eq!(mean(&t), 2.5);
        assert_eq!(prod(&t), 24.0);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), 4.0);
        assert_eq!(ptp(&t), 3.0);
        assert_eq!(argmax(&t), 3);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_variance_and_std() {
        let t = Tensor::from_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], vec![8]);
        let v = var(&t, 0); // Population variance
        assert_eq!(v, 4.0);
        assert_eq!(std(&t, 0), 2.0);
    }

    #[test]
    fn test_logsumexp_stability() {
        let t = Tensor::from_slice(&[1000.0, 1000.0], vec![2]);
        let lse = log_sum_exp(&t);
        assert!((lse - (1000.0 + (2.0f64).ln())).abs() < 1e-6);
    }

    #[test]
    fn test_boolean_reductions() {
        let all_true = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        assert!(all(&all_true));
        assert!(any(&all_true));

        let has_zero = Tensor::from_slice(&[1.0, 0.0, 3.0], vec![3]);
        assert!(!all(&has_zero));
        assert!(any(&has_zero));
    }

    #[test]
    fn test_nan_reductions() {
        let t = Tensor::from_slice(&[1.0, f64::NAN, 3.0], vec![3]);
        assert_eq!(nansum(&t), 4.0);
        assert_eq!(nanmean(&t), 2.0);
        assert_eq!(nanmin(&t), 1.0);
        assert_eq!(nanmax(&t), 3.0);
    }

    #[test]
    fn test_reduction_stress_case_001() {
        let count = (1 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_002() {
        let count = (2 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_003() {
        let count = (3 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_004() {
        let count = (4 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_005() {
        let count = (5 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_006() {
        let count = (6 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_007() {
        let count = (7 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_008() {
        let count = (8 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_009() {
        let count = (9 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_010() {
        let count = (10 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_011() {
        let count = (11 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_012() {
        let count = (12 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_013() {
        let count = (13 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_014() {
        let count = (14 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_015() {
        let count = (15 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_016() {
        let count = (16 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_017() {
        let count = (17 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_018() {
        let count = (18 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_019() {
        let count = (19 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_020() {
        let count = (20 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_021() {
        let count = (21 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_022() {
        let count = (22 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_023() {
        let count = (23 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_024() {
        let count = (24 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_025() {
        let count = (25 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_026() {
        let count = (26 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_027() {
        let count = (27 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_028() {
        let count = (28 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_029() {
        let count = (29 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_030() {
        let count = (30 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_031() {
        let count = (31 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_032() {
        let count = (32 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_033() {
        let count = (33 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_034() {
        let count = (34 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_035() {
        let count = (35 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_036() {
        let count = (36 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_037() {
        let count = (37 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_038() {
        let count = (38 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_039() {
        let count = (39 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_040() {
        let count = (40 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_041() {
        let count = (41 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_042() {
        let count = (42 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_043() {
        let count = (43 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_044() {
        let count = (44 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_045() {
        let count = (45 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_046() {
        let count = (46 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_047() {
        let count = (47 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_048() {
        let count = (48 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_049() {
        let count = (49 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_050() {
        let count = (50 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_051() {
        let count = (51 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_052() {
        let count = (52 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_053() {
        let count = (53 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_054() {
        let count = (54 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_055() {
        let count = (55 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_056() {
        let count = (56 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_057() {
        let count = (57 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_058() {
        let count = (58 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_059() {
        let count = (59 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_060() {
        let count = (60 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_061() {
        let count = (61 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_062() {
        let count = (62 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_063() {
        let count = (63 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_064() {
        let count = (64 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_065() {
        let count = (65 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_066() {
        let count = (66 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_067() {
        let count = (67 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_068() {
        let count = (68 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_069() {
        let count = (69 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_070() {
        let count = (70 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_071() {
        let count = (71 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_072() {
        let count = (72 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_073() {
        let count = (73 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_074() {
        let count = (74 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_075() {
        let count = (75 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_076() {
        let count = (76 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_077() {
        let count = (77 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_078() {
        let count = (78 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_079() {
        let count = (79 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_080() {
        let count = (80 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_081() {
        let count = (81 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_082() {
        let count = (82 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_083() {
        let count = (83 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_084() {
        let count = (84 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_085() {
        let count = (85 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_086() {
        let count = (86 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_087() {
        let count = (87 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_088() {
        let count = (88 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_089() {
        let count = (89 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_090() {
        let count = (90 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_091() {
        let count = (91 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_092() {
        let count = (92 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_093() {
        let count = (93 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_094() {
        let count = (94 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_095() {
        let count = (95 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_096() {
        let count = (96 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_097() {
        let count = (97 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_098() {
        let count = (98 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_099() {
        let count = (99 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_100() {
        let count = (100 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_101() {
        let count = (101 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_102() {
        let count = (102 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_103() {
        let count = (103 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_104() {
        let count = (104 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_105() {
        let count = (105 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_106() {
        let count = (106 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_107() {
        let count = (107 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_108() {
        let count = (108 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_109() {
        let count = (109 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_110() {
        let count = (110 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_111() {
        let count = (111 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_112() {
        let count = (112 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_113() {
        let count = (113 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_114() {
        let count = (114 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_115() {
        let count = (115 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_116() {
        let count = (116 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_117() {
        let count = (117 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_118() {
        let count = (118 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_119() {
        let count = (119 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_120() {
        let count = (120 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_121() {
        let count = (121 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_122() {
        let count = (122 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_123() {
        let count = (123 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_124() {
        let count = (124 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_125() {
        let count = (125 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_126() {
        let count = (126 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_127() {
        let count = (127 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_128() {
        let count = (128 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_129() {
        let count = (129 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_130() {
        let count = (130 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_131() {
        let count = (131 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_132() {
        let count = (132 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_133() {
        let count = (133 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_134() {
        let count = (134 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_135() {
        let count = (135 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_136() {
        let count = (136 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_137() {
        let count = (137 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_138() {
        let count = (138 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_139() {
        let count = (139 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_140() {
        let count = (140 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_141() {
        let count = (141 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_142() {
        let count = (142 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_143() {
        let count = (143 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_144() {
        let count = (144 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_145() {
        let count = (145 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_146() {
        let count = (146 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_147() {
        let count = (147 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_148() {
        let count = (148 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_149() {
        let count = (149 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_150() {
        let count = (150 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_151() {
        let count = (151 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_152() {
        let count = (152 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_153() {
        let count = (153 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_154() {
        let count = (154 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_155() {
        let count = (155 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_156() {
        let count = (156 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_157() {
        let count = (157 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_158() {
        let count = (158 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_159() {
        let count = (159 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_160() {
        let count = (160 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_161() {
        let count = (161 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_162() {
        let count = (162 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_163() {
        let count = (163 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_164() {
        let count = (164 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_165() {
        let count = (165 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_166() {
        let count = (166 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_167() {
        let count = (167 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_168() {
        let count = (168 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_169() {
        let count = (169 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_170() {
        let count = (170 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_171() {
        let count = (171 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_172() {
        let count = (172 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_173() {
        let count = (173 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_174() {
        let count = (174 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_175() {
        let count = (175 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_176() {
        let count = (176 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_177() {
        let count = (177 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_178() {
        let count = (178 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_179() {
        let count = (179 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_180() {
        let count = (180 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_181() {
        let count = (181 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_182() {
        let count = (182 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_183() {
        let count = (183 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_184() {
        let count = (184 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_185() {
        let count = (185 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_186() {
        let count = (186 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_187() {
        let count = (187 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_188() {
        let count = (188 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_189() {
        let count = (189 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_190() {
        let count = (190 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_191() {
        let count = (191 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_192() {
        let count = (192 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_193() {
        let count = (193 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_194() {
        let count = (194 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_195() {
        let count = (195 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_196() {
        let count = (196 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_197() {
        let count = (197 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_198() {
        let count = (198 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_199() {
        let count = (199 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_200() {
        let count = (200 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_201() {
        let count = (201 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_202() {
        let count = (202 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_203() {
        let count = (203 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_204() {
        let count = (204 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_205() {
        let count = (205 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_206() {
        let count = (206 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_207() {
        let count = (207 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_208() {
        let count = (208 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_209() {
        let count = (209 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_210() {
        let count = (210 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_211() {
        let count = (211 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_212() {
        let count = (212 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_213() {
        let count = (213 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_214() {
        let count = (214 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_215() {
        let count = (215 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_216() {
        let count = (216 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_217() {
        let count = (217 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_218() {
        let count = (218 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_219() {
        let count = (219 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_220() {
        let count = (220 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_221() {
        let count = (221 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_222() {
        let count = (222 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_223() {
        let count = (223 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_224() {
        let count = (224 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_225() {
        let count = (225 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_226() {
        let count = (226 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_227() {
        let count = (227 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_228() {
        let count = (228 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_229() {
        let count = (229 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_230() {
        let count = (230 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_231() {
        let count = (231 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_232() {
        let count = (232 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_233() {
        let count = (233 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_234() {
        let count = (234 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_235() {
        let count = (235 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_236() {
        let count = (236 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_237() {
        let count = (237 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_238() {
        let count = (238 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_239() {
        let count = (239 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_240() {
        let count = (240 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_241() {
        let count = (241 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_242() {
        let count = (242 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_243() {
        let count = (243 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }

    #[test]
    fn test_reduction_stress_case_244() {
        let count = (244 % 16) + 2;
        let vals: Vec<f64> = (1..=count).map(|i| i as f64).collect();
        let t = Tensor::from_slice(&vals, vec![count]);
        let expected_sum = (count * (count + 1) / 2) as f64;
        assert_eq!(sum(&t), expected_sum);
        assert_eq!(min(&t), 1.0);
        assert_eq!(max(&t), count as f64);
        assert_eq!(argmax(&t), count - 1);
        assert_eq!(argmin(&t), 0);
    }
}
