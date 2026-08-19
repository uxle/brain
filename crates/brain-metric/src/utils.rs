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
}
