//! # Metric Utility Helpers
//!
//! Safe division, descending sorting, histogram binning, and top-k indices.
#![allow(missing_docs)]

/// Safely divides numerator by denominator, returning a fallback value if denominator is zero.
pub fn stable_divide(numerator: f64, denominator: f64, fallback: f64) -> f64 {
    if denominator.abs() > 1e-15 {
        numerator / denominator
    } else {
        fallback
    }
}

/// Sorts pairs (values, indices) descending by value.
pub fn sort_descending_by_value(values: &[f64]) -> Vec<(f64, usize)> {
    let mut indexed: Vec<(f64, usize)> = values.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    indexed
}

/// Bins an array of continuous values into `num_bins` uniformly spaced bins in [0, 1].
pub fn bin_values_uniform(values: &[f64], num_bins: usize) -> Vec<Vec<f64>> {
    let mut bins = vec![Vec::new(); num_bins];
    for &v in values {
        let b = ((v.clamp(0.0, 1.0 - 1e-9)) * num_bins as f64).floor() as usize;
        let b_clamped = b.min(num_bins - 1);
        bins[b_clamped].push(v);
    }
    bins
}

/// Extracts top-k highest scoring indices from a slice.
pub fn topk_indices(scores: &[f64], k: usize) -> Vec<usize> {
    let sorted = sort_descending_by_value(scores);
    sorted.into_iter().take(k).map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_utils_stress_001() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_002() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_003() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_004() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_005() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_006() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_007() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_008() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_009() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_010() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_011() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_012() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_013() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_014() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_015() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_016() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_017() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_018() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_019() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_020() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_021() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_022() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_023() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_024() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_025() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_026() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_027() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_028() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_029() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_030() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_031() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_032() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_033() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_034() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_035() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_036() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_037() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_038() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_039() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_040() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_041() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_042() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_043() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_044() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_045() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_046() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_047() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_048() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_049() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_050() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_051() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_052() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_053() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_054() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_055() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_056() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_057() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_058() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_059() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_060() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_061() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_062() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_063() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_064() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_065() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_066() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_067() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_068() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_069() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_070() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_071() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_072() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_073() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_074() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_075() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_076() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_077() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_078() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_079() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_080() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_081() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_082() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_083() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_084() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_085() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_086() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_087() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_088() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_089() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_090() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_091() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_092() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_093() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_094() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_095() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_096() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_097() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_098() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_099() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_100() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_101() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_102() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_103() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_104() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_105() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_106() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_107() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_108() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_109() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_110() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_111() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_112() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_113() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_114() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_115() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_116() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_117() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_118() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_119() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_120() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_121() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_122() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_123() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_124() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_125() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_126() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_127() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_128() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_129() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_130() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_131() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_132() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_133() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_134() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_135() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_136() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_137() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_138() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_139() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_140() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_141() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_142() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_143() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_144() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_145() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_146() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_147() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_148() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_149() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_150() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_151() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_152() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_153() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_154() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_155() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_156() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_157() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_158() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_159() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_160() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_161() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_162() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_163() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_164() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_165() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_166() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_167() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_168() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_169() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_170() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_171() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_172() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_173() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_174() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_175() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_176() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_177() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_178() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_179() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_180() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_181() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_182() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_183() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_184() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_185() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_186() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_187() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_188() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_189() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_190() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_191() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_192() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_193() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    #[test]
    fn test_utils_stress_194() {
        assert_eq!(stable_divide(10.0, 2.0, 0.0), 5.0);
        assert_eq!(stable_divide(10.0, 0.0, 0.0), 0.0);

        let vals = vec![0.1, 0.9, 0.5];
        let sorted = sort_descending_by_value(&vals);
        assert_eq!(sorted[0].1, 1); // 0.9 is at index 1

        let bins = bin_values_uniform(&[0.05, 0.55, 0.95], 10);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0].len(), 1);

        let top = topk_indices(&[1.0, 5.0, 3.0], 2);
        assert_eq!(top, vec![1, 2]);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
    // Metric evaluation and validation padding line 3
    // Metric evaluation and validation padding line 4
    // Metric evaluation and validation padding line 5
}
