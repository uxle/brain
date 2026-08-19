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
// Variance / Standard Deviation Along a Dimension
// =============================================================================

/// Computes variance along a specified dimension with the given degrees-of-freedom correction.
pub fn var_along_dim(a: &Tensor, dim: usize, keepdim: bool, correction: usize) -> Tensor {
    assert!(dim < a.ndim(), "var_along_dim: dim out of bounds");
    let n = a.shape()[dim];
    assert!(n > correction, "var_along_dim: degrees of freedom <= 0");
    let mean_t = mean_along_dim(a, dim, true);
    let sq_diff = a.map2(&mean_t, |m, x| (x - m) * (x - m));
    let sum_sq = reduce_along_dim(&sq_diff, dim, keepdim, 0.0, |acc, x| acc + x);
    crate::tensor::arithmetic::mul_scalar(&sum_sq, 1.0 / ((n - correction) as f64))
}

/// Computes standard deviation along a specified dimension.
pub fn std_along_dim(a: &Tensor, dim: usize, keepdim: bool, correction: usize) -> Tensor {
    let v = var_along_dim(a, dim, keepdim, correction);
    crate::tensor::math::sqrt(&v)
}

/// Computes the mean and (unbiased) variance together, reusing the single mean pass.
pub fn var_mean(a: &Tensor, correction: usize) -> (f64, f64) {
    assert!(a.numel() > correction, "var_mean: degrees of freedom <= 0");
    let m = mean(a);
    let sum_sq_diff: f64 = a.data().iter().map(|&x| (x - m).powi(2)).sum();
    let v = sum_sq_diff / ((a.numel() - correction) as f64);
    (v, m)
}

// =============================================================================
// Cumulative Reductions
// =============================================================================

/// Computes the inclusive cumulative sum along a dimension.
pub fn cumsum(a: &Tensor, dim: usize) -> Tensor {
    assert!(dim < a.ndim(), "cumsum: dim out of bounds");
    let mut acc = vec![0.0; a.numel() / a.shape()[dim]];
    let mut out_data = vec![0.0; a.numel()];
    let mut coords = vec![0usize; a.ndim()];

    for flat in 0..a.numel() {
        let val = a.get(flat);
        let mut stripped = coords.clone();
        stripped.remove(dim);
        let key = flat_index(&stripped, &a.shape().iter().enumerate().filter(|(d, _)| *d != dim).map(|(_, &s)| s).collect::<Vec<_>>());
        acc[key] += val;
        out_data[flat] = acc[key];

        for d in (0..a.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < a.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Tensor::new(out_data, a.shape().to_vec())
}

/// Computes the inclusive cumulative product along a dimension.
pub fn cumprod(a: &Tensor, dim: usize) -> Tensor {
    assert!(dim < a.ndim(), "cumprod: dim out of bounds");
    let mut acc = vec![1.0; a.numel() / a.shape()[dim]];
    let mut out_data = vec![0.0; a.numel()];
    let mut coords = vec![0usize; a.ndim()];

    for flat in 0..a.numel() {
        let val = a.get(flat);
        let mut stripped = coords.clone();
        stripped.remove(dim);
        let key = flat_index(&stripped, &a.shape().iter().enumerate().filter(|(d, _)| *d != dim).map(|(_, &s)| s).collect::<Vec<_>>());
        acc[key] *= val;
        out_data[flat] = acc[key];

        for d in (0..a.ndim()).rev() {
            coords[d] += 1;
            if coords[d] < a.shape()[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Tensor::new(out_data, a.shape().to_vec())
}

/// Computes the row-major flat index of `coords` given `shape`.
fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0usize;
    for (c, &s) in coords.iter().zip(shape) {
        idx = idx * s + c;
    }
    idx
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
    fn test_reductions_along_dim_and_global() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(sum(&a), 21.0);
        assert_eq!(mean(&a), 3.5);
        assert_eq!(min(&a), 1.0);
        assert_eq!(max(&a), 6.0);
        assert_eq!(prod(&a), 720.0);
    }

    #[test]
    fn test_var_along_dim() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // Column means: [2.5, 3.5, 4.5]; population variances: [2.25, 2.25, 2.25]
        let v = var_along_dim(&a, 0, false, 0);
        for i in 0..3 {
            assert!((v.get(i) - 2.25).abs() < 1e-9, "var along dim 0 mismatch at {i}");
        }
        // Row means: [2.0, 5.0]; population variances: [2.0/3, 2.0/3]
        let v = var_along_dim(&a, 1, false, 0);
        for i in 0..2 {
            assert!((v.get(i) - 2.0 / 3.0).abs() < 1e-9, "var along dim 1 mismatch at {i}");
        }
        // keepdim keeps the reduced axis
        let v = var_along_dim(&a, 1, true, 0);
        assert_eq!(v.shape(), &[2, 1]);
        // Sample correction: row sample variances = 1.0 for [1,2,3] and [4,5,6]
        let v = var_along_dim(&a, 1, false, 1);
        assert!((v.get(0) - 1.0).abs() < 1e-9);
        assert!((v.get(1) - 1.0).abs() < 1e-9);
        // std is sqrt of var
        let s = std_along_dim(&a, 1, false, 1);
        assert!((s.get(0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_var_mean_global() {
        let a = Tensor::from_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], vec![8]);
        let (v, m) = var_mean(&a, 0);
        assert_eq!(m, 5.0);
        assert_eq!(v, 4.0);
    }

    #[test]
    fn test_cumsum_cumprod() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let cs = cumsum(&a, 1);
        assert_eq!(cs.to_vec(), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
        let cs = cumsum(&a, 0);
        assert_eq!(cs.to_vec(), vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
        let cp = cumprod(&a, 1);
        assert_eq!(cp.to_vec(), vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
        // Negative values
        let b = Tensor::from_slice(&[-2.0, 3.0, -1.0], vec![3]);
        let cp = cumprod(&b, 0);
        assert_eq!(cp.to_vec(), vec![-2.0, -6.0, 6.0]);
    }
}
