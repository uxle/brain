//! Histogram calculations, bincounts, quantiles, and empirical distributions.
//!
//! This module provides statistical histograms, bincount frequencies, percentiles, and median computations.

use crate::tensor::Tensor;

/// Computes the histogram of tensor values over a specified range.
///
/// Returns (bin_counts, bin_edges).
pub fn histogram(input: &Tensor, bins: usize, range: (f64, f64)) -> (Tensor, Tensor) {
    assert!(bins > 0);
    let (min_val, max_val) = range;
    assert!(min_val < max_val);

    let bin_width = (max_val - min_val) / (bins as f64);
    let mut counts = vec![0.0; bins];
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(min_val + (i as f64) * bin_width);
    }

    for &x in input.data() {
        if x >= min_val && x <= max_val {
            let mut b = ((x - min_val) / bin_width) as usize;
            if b >= bins {
                b = bins - 1;
            }
            counts[b] += 1.0;
        }
    }

    (
        Tensor::new(counts, vec![bins]),
        Tensor::new(edges, vec![bins + 1]),
    )
}

/// Counts the number of occurrences of each non-negative integer value in an array.
pub fn bincount(input: &Tensor, minlength: usize) -> Tensor {
    if input.is_empty() {
        return Tensor::zeros(vec![minlength]);
    }
    let mut max_val = 0usize;
    for &x in input.data() {
        assert!(
            x >= 0.0,
            "bincount: input values must be non-negative integers"
        );
        max_val = max_val.max(x as usize);
    }
    let len = (max_val + 1).max(minlength);
    let mut counts = vec![0.0; len];
    for &x in input.data() {
        counts[x as usize] += 1.0;
    }
    Tensor::new(counts, vec![len])
}

/// Computes the q-th quantile (0.0 <= q <= 1.0) of tensor elements.
pub fn quantile(input: &Tensor, q: f64) -> f64 {
    assert!((0.0..=1.0).contains(&q));
    assert!(!input.is_empty());
    let mut sorted = input.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

/// Computes the median of tensor elements.
pub fn median(input: &Tensor) -> f64 {
    quantile(input, 0.5)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let t = Tensor::from_slice(&[0.5, 1.5, 2.5, 3.5], vec![4]);
        let (counts, edges) = histogram(&t, 4, (0.0, 4.0));
        assert_eq!(counts.data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(edges.numel(), 5);
    }

    #[test]
    fn test_bincount_and_median() {
        let t = Tensor::from_slice(&[0.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![6]);
        let bc = bincount(&t, 0);
        assert_eq!(bc.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(median(&t), 2.0);
    }

    #[test]
    fn test_hist_bincount_median() {
        let t = Tensor::from_slice(&[1.0, 2.0, 2.0, 3.0, 3.0, 3.0], vec![6]);
        let bc = bincount(&t, 0);
        assert_eq!(bc.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);

        let med = median(&t);
        assert_eq!(med, 3.0);
    }
}
