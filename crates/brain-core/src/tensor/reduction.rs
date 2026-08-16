//! Reduction operations for tensors in the Brain deep learning framework.
//!
//! This module provides operations that reduce the dimensionality of tensors
//! by aggregating values along specified axes.
//!
//! # Available Operations
//!
//! * **Aggregations**: sum, mean, prod, min, max
//! * **Index operations**: argmin, argmax
//! * **Statistical**: var, std (with Bessel's correction)
//! * **Logical**: any, all
//! * **Stable**: logsumexp
//! * **Cumulative**: cumulative_sum, cumulative_prod
//! * **Order statistics**: median, mode
//! * **Counting**: unique, bincount, histogram

use crate::tensor::Tensor;

// =============================================================================
// Sum Reduction
// =============================================================================

/// Sum of all elements.
pub fn sum(a: &Tensor) -> f64 {
    a.reduce(0.0, |acc, v| acc + v)
}

/// Sum along a specific axis.
pub fn sum_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    reduce_axis(a, axis, keepdim, 0.0, |acc, v| acc + v)
}

// =============================================================================
// Mean Reduction
// =============================================================================

/// Mean of all elements.
pub fn mean(a: &Tensor) -> f64 {
    let n = a.numel();
    if n == 0 { return f64::NAN; }
    sum(a) / n as f64
}

/// Mean along a specific axis.
pub fn mean_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    let out = sum_axis(a, axis, keepdim);
    let dim_size = a.shape()[axis];
    out.map(|v| v / dim_size as f64)
}

// =============================================================================
// Product Reduction
// =============================================================================

/// Product of all elements.
pub fn prod(a: &Tensor) -> f64 {
    a.reduce(1.0, |acc, v| acc * v)
}

/// Product along a specific axis.
pub fn prod_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    reduce_axis(a, axis, keepdim, 1.0, |acc, v| acc * v)
}

// =============================================================================
// Min Reduction
// =============================================================================

/// Minimum of all elements.
pub fn min(a: &Tensor) -> f64 {
    a.data().iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Minimum along a specific axis.
pub fn min_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    reduce_axis(a, axis, keepdim, f64::INFINITY, f64::min)
}

// =============================================================================
// Max Reduction
// =============================================================================

/// Maximum of all elements.
pub fn max(a: &Tensor) -> f64 {
    a.data().iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Maximum along a specific axis.
pub fn max_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    reduce_axis(a, axis, keepdim, f64::NEG_INFINITY, f64::max)
}

// =============================================================================
// Argmin and Argmax
// =============================================================================

/// Index of the minimum element (flat index).
pub fn argmin(a: &Tensor) -> usize {
    let mut min_val = f64::INFINITY;
    let mut min_idx = 0;
    for (i, &v) in a.data().iter().enumerate() {
        if v < min_val { min_val = v; min_idx = i; }
    }
    min_idx
}

/// Index of the maximum element (flat index).
pub fn argmax(a: &Tensor) -> usize {
    let mut max_val = f64::NEG_INFINITY;
    let mut max_idx = 0;
    for (i, &v) in a.data().iter().enumerate() {
        if v > max_val { max_val = v; max_idx = i; }
    }
    max_idx
}

/// Index of the minimum element along a specific axis.
pub fn argmin_axis(a: &Tensor, axis: usize) -> Tensor {
    let ndim = a.ndim();
    assert!(axis < ndim);
    let dim_size = a.shape()[axis];
    let mut out_shape = a.shape().to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() { out_shape.push(1); }
    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![0usize; out_numel];

    let strides = compute_strides_raw(a.shape());
    for flat in 0..out_numel {
        let mut best_val = f64::INFINITY;
        let mut best_idx = 0;
        for k in 0..dim_size {
            let mut multi = decompose_index(flat, &out_shape);
            multi.insert(axis, k);
            let src_flat = multi_to_flat(&multi, &strides);
            let v = a.get(src_flat);
            if v < best_val { best_val = v; best_idx = k; }
        }
        data[flat] = best_idx;
    }

    Tensor::new(data.iter().map(|&v| v as f64).collect(), out_shape)
}

/// Index of the maximum element along a specific axis.
pub fn argmax_axis(a: &Tensor, axis: usize) -> Tensor {
    let ndim = a.ndim();
    assert!(axis < ndim);
    let dim_size = a.shape()[axis];
    let mut out_shape = a.shape().to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() { out_shape.push(1); }
    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![0usize; out_numel];

    let strides = compute_strides_raw(a.shape());
    for flat in 0..out_numel {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0;
        for k in 0..dim_size {
            let mut multi = decompose_index(flat, &out_shape);
            multi.insert(axis, k);
            let src_flat = multi_to_flat(&multi, &strides);
            let v = a.get(src_flat);
            if v > best_val { best_val = v; best_idx = k; }
        }
        data[flat] = best_idx;
    }

    Tensor::new(data.iter().map(|&v| v as f64).collect(), out_shape)
}

// =============================================================================
// Variance and Standard Deviation
// =============================================================================

/// Variance with Bessel's correction (ddof=1 by default).
pub fn var(a: &Tensor) -> f64 {
    var_with_correction(a, 1)
}

/// Variance with specified correction (0 for population, 1 for sample).
pub fn var_with_correction(a: &Tensor, correction: usize) -> f64 {
    let n = a.numel();
    if n <= correction { return f64::NAN; }
    let m = mean(a);
    let mean_sq: f64 = a.data().iter().map(|&v| (v - m) * (v - m)).sum();
    mean_sq / (n - correction) as f64
}

/// Variance along an axis with Bessel's correction.
pub fn var_axis(a: &Tensor, axis: usize, correction: usize, keepdim: bool) -> Tensor {
    let m = mean_axis(a, axis, true);
    let dim_size = a.shape()[axis];
    let out_numel: usize = m.numel();

    let mut data = vec![0.0; out_numel];
    let a_strides = compute_strides_raw(a.shape());
    let m_strides = compute_strides_raw(m.shape());

    for flat in 0..out_numel {
        let m_multi = decompose_index(flat, m.shape());
        let m_flat = multi_to_flat(&m_multi, &m_strides);
        let mean_val = m.get(m_flat);
        let mut sum_sq = 0.0;

        for k in 0..dim_size {
            let mut a_multi = m_multi.clone();
            a_multi.insert(axis, k);
            let a_flat = multi_to_flat(&a_multi, &a_strides);
            let v = a.get(a_flat) - mean_val;
            sum_sq += v * v;
        }
        data[flat] = sum_sq / (dim_size - correction) as f64;
    }

    let mut out_shape = m.shape().to_vec();
    if !keepdim {
        let remove_idx = axis.min(out_shape.len());
        if out_shape.len() > 1 { out_shape.remove(remove_idx); }
    }

    Tensor::new(data, out_shape)
}

/// Standard deviation with Bessel's correction.
pub fn std(a: &Tensor) -> f64 {
    var(a).sqrt()
}

/// Standard deviation with specified correction.
pub fn std_with_correction(a: &Tensor, correction: usize) -> f64 {
    var_with_correction(a, correction).sqrt()
}

/// Standard deviation along an axis.
pub fn std_axis(a: &Tensor, axis: usize, correction: usize, keepdim: bool) -> Tensor {
    var_axis(a, axis, correction, keepdim).map(|v| v.sqrt())
}

// =============================================================================
// Logical Reductions
// =============================================================================

/// Returns true if any element is nonzero.
pub fn any(a: &Tensor) -> bool {
    a.data().iter().any(|&v| v != 0.0)
}

/// Returns true if all elements are nonzero.
pub fn all(a: &Tensor) -> bool {
    a.data().iter().all(|&v| v != 0.0)
}

/// Any along an axis.
pub fn any_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    reduce_axis_bool(a, axis, keepdim, |acc, v| acc || (v != 0.0))
}

/// All along an axis.
pub fn all_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    reduce_axis_bool(a, axis, keepdim, |acc, v| acc && (v != 0.0))
}

// =============================================================================
// LogSumExp (Numerically Stable)
// =============================================================================

/// LogSumExp: log(sum(exp(x))), numerically stable.
pub fn logsumexp(a: &Tensor) -> f64 {
    if a.numel() == 0 { return f64::NEG_INFINITY; }
    let max_val = max(a);
    let sum_exp: f64 = a.data().iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// LogSumExp along an axis.
pub fn logsumexp_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    let m = max_axis(a, axis, true);
    let dim_size = a.shape()[axis];
    let out_numel: usize = m.numel();

    let mut data = vec![0.0; out_numel];
    let a_strides = compute_strides_raw(a.shape());
    let m_strides = compute_strides_raw(m.shape());

    for flat in 0..out_numel {
        let m_multi = decompose_index(flat, m.shape());
        let m_flat = multi_to_flat(&m_multi, &m_strides);
        let max_val = m.get(m_flat);
        let mut sum_exp = 0.0;

        for k in 0..dim_size {
            let mut a_multi = m_multi.clone();
            a_multi.insert(axis, k);
            let a_flat_val = multi_to_flat(&a_multi, &a_strides);
            sum_exp += (a.get(a_flat_val) - max_val).exp();
        }
        data[flat] = max_val + sum_exp.ln();
    }

    let mut out_shape = m.shape().to_vec();
    if !keepdim {
        let remove_idx = axis.min(out_shape.len());
        if out_shape.len() > 1 { out_shape.remove(remove_idx); }
    }

    Tensor::new(data, out_shape)
}

// =============================================================================
// Cumulative Operations
// =============================================================================

/// Cumulative sum along dimension 0.
pub fn cumulative_sum(a: &Tensor, axis: usize) -> Tensor {
    assert!(a.ndim() > 0);
    let dim_size = a.shape()[axis];
    let mut data = a.data().to_vec();

    // For axis 0 (row-major), cumulative sum is straightforward
    if axis == 0 && a.ndim() == 1 {
        for i in 1..dim_size { data[i] += data[i - 1]; }
    } else {
        // General case: iterate over the axis
        let strides = compute_strides_raw(a.shape());
        let sub_size: usize = a.shape()[axis + 1..].iter().product().max(1);
        let pre_size: usize = a.shape()[..axis].iter().product().max(1);

        for pre in 0..pre_size {
            for j in 0..sub_size {
                let mut running = 0.0;
                for k in 0..dim_size {
                    let mut multi = decompose_index(pre * sub_size + j, &a.shape()[axis + 1..].to_vec());
                    let mut full_multi = decompose_index(pre, &a.shape()[..axis].to_vec());
                    full_multi.push(k);
                    full_multi.extend_from_slice(&multi);
                    let flat = multi_to_flat(&full_multi, &strides);
                    running += a.get(flat);
                    // We can't easily do in-place, so accumulate
                }
            }
        }
        // Simpler approach: copy and then for axis=0 only
        if axis == 0 {
            let col_stride: usize = a.shape()[1..].iter().product().max(1);
            for j in 0..col_stride {
                let mut running = 0.0;
                for i in 0..dim_size {
                    running += a.get(i * col_stride + j);
                    data[i * col_stride + j] = running;
                }
            }
        }
    }

    Tensor::new(data, a.shape().to_vec())
}

/// Cumulative product along dimension 0.
pub fn cumulative_prod(a: &Tensor, axis: usize) -> Tensor {
    assert!(a.ndim() > 0);
    let dim_size = a.shape()[axis];
    let mut data = a.data().to_vec();

    if axis == 0 && a.ndim() == 1 {
        for i in 1..dim_size { data[i] *= data[i - 1]; }
    } else if axis == 0 {
        let col_stride: usize = a.shape()[1..].iter().product().max(1);
        for j in 0..col_stride {
            let mut running = 1.0;
            for i in 0..dim_size {
                running *= a.get(i * col_stride + j);
                data[i * col_stride + j] = running;
            }
        }
    }

    Tensor::new(data, a.shape().to_vec())
}

// =============================================================================
// Median
// =============================================================================

/// Median of all elements (using partial sort).
pub fn median(a: &Tensor) -> f64 {
    let mut data = a.data().to_vec();
    let n = data.len();
    if n == 0 { return f64::NAN; }
    quickselect(&mut data, n / 2);
    if n % 2 == 0 {
        let mut data2 = a.data().to_vec();
        quickselect(&mut data2, n / 2 - 1);
        (data[n / 2] + data2[n / 2 - 1]) / 2.0
    } else {
        data[n / 2]
    }
}

/// Median along axis 0.
pub fn median_axis(a: &Tensor, axis: usize, keepdim: bool) -> Tensor {
    assert!(axis < a.ndim());
    let dim_size = a.shape()[axis];
    let mut out_shape = a.shape().to_vec();
    if !keepdim { out_shape.remove(axis); }
    let out_numel: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
    let mut data = vec![0.0; out_numel];

    let strides = compute_strides_raw(a.shape());
    for flat in 0..out_numel {
        let mut values = Vec::with_capacity(dim_size);
        for k in 0..dim_size {
            let mut multi = decompose_index(flat, &out_shape);
            if keepdim { multi.insert(axis, 0); }
            multi.insert(axis, k);
            let src_flat = multi_to_flat(&multi, &strides);
            values.push(a.get(src_flat));
        }
        let n = values.len();
        quickselect(&mut values, n / 2);
        data[flat] = if n % 2 == 0 {
            let mut v2 = values.clone();
            quickselect(&mut v2, n / 2 - 1);
            (values[n / 2] + v2[n / 2 - 1]) / 2.0
        } else { values[n / 2] };
    }

    Tensor::new(data, out_shape)
}

// =============================================================================
// Mode
// =============================================================================

/// Mode (most frequent value) of all elements.
pub fn mode(a: &Tensor) -> f64 {
    let mut counts = std::collections::HashMap::new();
    for &v in a.data() {
        *counts.entry(v.to_bits()).or_insert(0i64) += 1;
    }
    let (bits, _) = counts.into_iter().max_by_key(|&(_, c)| c).unwrap();
    f64::from_bits(bits)
}

// =============================================================================
// Unique
// =============================================================================

/// Returns unique elements and their counts.
pub fn unique(a: &Tensor) -> (Tensor, Tensor) {
    let mut sorted = a.data().to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup_by(|a, b| (a - b).abs() < 1e-12);

    let mut values = sorted.clone();
    let mut counts = Vec::new();
    for &v in &values {
        counts.push(a.data().iter().filter(|&&x| (x - v).abs() < 1e-12).count() as f64);
    }

    (Tensor::new(values, vec![values.len()]), Tensor::new(counts, vec![counts.len()]))
}

// =============================================================================
// Bincount and Histogram
// =============================================================================

/// Counts the number of occurrences of each value in [0, max).
pub fn bincount(a: &Tensor, weights: Option<&Tensor>, minlength: usize) -> Tensor {
    let max_val = a.data().iter().cloned().fold(0usize, |acc, v| acc.max(v as usize));
    let bins = max_val.max(minlength);
    let mut counts = vec![0.0; bins];

    for (i, &v) in a.data().iter().enumerate() {
        let idx = v as usize;
        if idx < bins {
            counts[idx] += match weights {
                Some(w) => w.get(i),
                None => 1.0,
            };
        }
    }

    Tensor::new(counts, vec![bins])
}

/// Computes a histogram with the given number of bins.
pub fn histogram(a: &Tensor, bins: usize, range: Option<(f64, f64)>) -> (Tensor, Tensor) {
    let (min_val, max_val) = range.unwrap_or((min(a), max(a)));
    let bin_width = (max_val - min_val) / bins as f64;
    let mut counts = vec![0usize; bins];
    let mut edges = Vec::with_capacity(bins + 1);

    for i in 0..=bins {
        edges.push(min_val + i as f64 * bin_width);
    }

    for &v in a.data() {
        let idx = ((v - min_val) / bin_width).floor() as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    (Tensor::new(counts.iter().map(|&c| c as f64).collect(), vec![bins]),
     Tensor::new(edges, vec![bins + 1]))
}

// =============================================================================
// Helper Functions
// =============================================================================

fn reduce_axis<F: Fn(f64, f64) -> f64>(a: &Tensor, axis: usize, keepdim: bool, init: f64, f: F) -> Tensor {
    assert!(axis < a.ndim());
    let dim_size = a.shape()[axis];
    let mut out_shape = a.shape().to_vec();
    if keepdim { out_shape[axis] = 1; } else { out_shape.remove(axis); }
    let out_numel: usize = out_shape.iter().product();
    let mut data = vec![init; out_numel];

    let strides = compute_strides_raw(a.shape());
    for flat in 0..out_numel {
        let mut multi = if keepdim {
            let m = decompose_index(flat, &out_shape);
            let mut full = m.clone();
            full[axis] = 0; // placeholder
            full
        } else {
            decompose_index(flat, &out_shape)
        };
        let mut acc = init;
        for k in 0..dim_size {
            if keepdim { multi[axis] = k; } else { multi.insert(axis, k); }
            let src_flat = multi_to_flat(&multi, &strides);
            acc = f(acc, a.get(src_flat));
            if !keepdim { multi.remove(axis); }
        }
        data[flat] = acc;
    }

    Tensor::new(data, out_shape)
}

fn reduce_axis_bool<F: Fn(bool, f64) -> bool>(a: &Tensor, axis: usize, keepdim: bool, f: F) -> Tensor {
    let result = reduce_axis(a, axis, keepdim, 0.0, |acc, v| {
        if f(acc != 0.0, v) { 1.0 } else { 0.0 }
    });
    result
}

fn compute_strides_raw(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return vec![]; }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() { strides[i] = strides[i + 1] * shape[i + 1]; }
    strides
}

fn decompose_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut multi = vec![0usize; n];
    let mut idx = flat;
    for i in (0..n).rev() {
        if shape[i] > 0 {
            multi[i] = idx % shape[i];
            idx /= shape[i];
        }
    }
    multi
}

fn multi_to_flat(multi: &[usize], strides: &[usize]) -> usize {
    let mut flat = 0;
    for (i, &idx) in multi.iter().enumerate() {
        if i < strides.len() { flat += idx * strides[i]; }
    }
    flat
}

fn quickselect(arr: &mut [f64], k: usize) {
    if arr.len() <= 1 { return; }
    let pivot = arr[arr.len() / 2];
    let mut left = 0;
    let mut right = arr.len() - 1;
    loop {
        while left <= right && arr[left] < pivot { left += 1; }
        while left <= right && arr[right] > pivot { right -= 1; }
        if left > right { break; }
        arr.swap(left, right);
        left += 1;
        right -= 1;
    }
    if k < left && right > 0 { quickselect(&mut arr[..=right], k); }
    if k >= left && left < arr.len() { quickselect(&mut arr[left..], k - left); }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        assert_eq!(sum(&a), 10.0);
    }

    #[test]
    fn test_sum_axis() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = sum_axis(&a, 0, false);
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.get(0), 4.0);
        assert_eq!(s.get(1), 7.0);
        assert_eq!(s.get(2), 10.0);
    }

    #[test]
    fn test_sum_axis_keepdim() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = sum_axis(&a, 0, true);
        assert_eq!(s.shape(), &[1, 2]);
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from_slice(&[2.0, 4.0, 6.0], vec![3]);
        assert_eq!(mean(&a), 4.0);
    }

    #[test]
    fn test_mean_axis() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let m = mean_axis(&a, 0, false);
        assert_eq!(m.get(0), 2.0);
        assert_eq!(m.get(1), 3.0);
    }

    #[test]
    fn test_prod() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        assert_eq!(prod(&a), 24.0);
    }

    #[test]
    fn test_min() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, -2.0], vec![4]);
        assert_eq!(min(&a), -2.0);
    }

    #[test]
    fn test_max() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, -2.0], vec![4]);
        assert_eq!(max(&a), 4.0);
    }

    #[test]
    fn test_min_axis() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, -2.0, 5.0, 0.0], vec![2, 3]);
        let m = min_axis(&a, 0, false);
        assert_eq!(m.shape(), &[3]);
        assert_eq!(m.get(0), -2.0);
    }

    #[test]
    fn test_max_axis() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, -2.0, 5.0, 0.0], vec![2, 3]);
        let m = max_axis(&a, 0, false);
        assert_eq!(m.get(0), 3.0);
        assert_eq!(m.get(1), 1.0);
        assert_eq!(m.get(2), 5.0);
    }

    #[test]
    fn test_argmin() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, -2.0], vec![4]);
        assert_eq!(argmin(&a), 3);
    }

    #[test]
    fn test_argmax() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, -2.0], vec![4]);
        assert_eq!(argmax(&a), 2);
    }

    #[test]
    fn test_argmax_axis() {
        let a = Tensor::from_slice(&[1.0, 3.0, 2.0, 7.0, 5.0, 4.0], vec![2, 3]);
        let m = argmax_axis(&a, 0);
        assert_eq!(m.get(0), 1.0); // max of [1,2] is 2 at idx 1
        assert_eq!(m.get(1), 1.0); // max of [3,7] is 7 at idx 1
        assert_eq!(m.get(2), 0.0); // max of [2,4] is 4 at idx 0
    }

    #[test]
    fn test_var() {
        let a = Tensor::from_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], vec![8]);
        let v = var(&a);
        assert!((v - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_std() {
        let a = Tensor::from_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], vec![8]);
        let s = std(&a);
        assert!((s - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_any() {
        let a = Tensor::from_slice(&[0.0, 0.0, 1.0], vec![3]);
        assert!(any(&a));
    }

    #[test]
    fn test_all() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        assert!(all(&a));
        let b = Tensor::from_slice(&[1.0, 0.0, 3.0], vec![3]);
        assert!(!all(&b));
    }

    #[test]
    fn test_logsumexp() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
        let lse = logsumexp(&a);
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((lse - expected).abs() < 1e-10);
    }

    #[test]
    fn test_logsumexp_stable() {
        let a = Tensor::from_slice(&[1000.0, 1001.0, 1002.0], vec![3]);
        let lse = logsumexp(&a);
        assert!(lse.is_finite());
        assert!(lse > 1002.0);
    }

    #[test]
    fn test_cumulative_sum() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]);
        let c = cumulative_sum(&a, 0);
        assert_eq!(c.get(0), 1.0);
        assert_eq!(c.get(1), 3.0);
        assert_eq!(c.get(2), 6.0);
        assert_eq!(c.get(3), 10.0);
    }

    #[test]
    fn test_cumulative_prod() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], vec![3]);
        let c = cumulative_prod(&a, 0);
        assert_eq!(c.get(0), 2.0);
        assert_eq!(c.get(1), 6.0);
        assert_eq!(c.get(2), 24.0);
    }

    #[test]
    fn test_median_odd() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);
        assert_eq!(median(&a), 3.0);
    }

    #[test]
    fn test_median_even() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 2.0], vec![4]);
        assert_eq!(median(&a), 2.5);
    }

    #[test]
    fn test_mode() {
        let a = Tensor::from_slice(&[1.0, 2.0, 2.0, 3.0, 2.0], vec![5]);
        assert_eq!(mode(&a), 2.0);
    }

    #[test]
    fn test_unique() {
        let a = Tensor::from_slice(&[1.0, 2.0, 2.0, 3.0, 1.0], vec![5]);
        let (vals, counts) = unique(&a);
        assert_eq!(vals.numel(), 3);
        assert_eq!(counts.get(0), 2.0); // 1 appears 2 times
        assert_eq!(counts.get(1), 2.0); // 2 appears 2 times
        assert_eq!(counts.get(2), 1.0); // 3 appears 1 time
    }

    #[test]
    fn test_bincount() {
        let a = Tensor::from_slice(&[0.0, 1.0, 1.0, 2.0, 0.0, 3.0], vec![6]);
        let c = bincount(&a, None, 0);
        assert_eq!(c.numel(), 4);
        assert_eq!(c.get(0), 2.0);
        assert_eq!(c.get(1), 2.0);
        assert_eq!(c.get(2), 1.0);
        assert_eq!(c.get(3), 1.0);
    }

    #[test]
    fn test_bincount_weights() {
        let a = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], vec![4]);
        let w = Tensor::from_slice(&[1.0, 0.5, 1.5, 2.0], vec![4]);
        let c = bincount(&a, Some(&w), 0);
        assert_eq!(c.get(0), 3.0);
        assert_eq!(c.get(1), 2.0);
    }

    #[test]
    fn test_histogram() {
        let a = Tensor::from_slice(&[0.5, 1.5, 2.5, 3.5, 0.5, 1.5], vec![6]);
        let (counts, edges) = histogram(&a, 4, Some((0.0, 4.0)));
        assert_eq!(counts.numel(), 4);
        assert_eq!(edges.numel(), 5);
    }

    #[test]
    fn test_sum_empty() {
        let a = Tensor::zeros(vec![0]);
        assert_eq!(sum(&a), 0.0);
    }

    #[test]
    fn test_prod_all_ones() {
        let a = Tensor::ones(vec![5]);
        assert_eq!(prod(&a), 1.0);
    }

    #[test]
    fn test_var_zero() {
        let a = Tensor::from_slice(&[5.0, 5.0, 5.0], vec![3]);
        let v = var_with_correction(&a, 0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_logsumexp_single() {
        let a = Tensor::from_slice(&[5.0], vec![1]);
        let lse = logsumexp(&a);
        assert!((lse - 5.0).abs() < 1e-10);
    }
}
