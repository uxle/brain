//! # Gradient Sparsification
//!
//! Top-K and threshold sparsification for communication efficiency.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for gradient sparsification.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub sparsity: f64,
}

impl Default for SparseConfig {
    fn default() -> Self { Self { sparsity: 0.9 } }
}

/// Returns a mask tensor with top-k fraction of elements kept.
pub fn top_k_sparsify(t: &Tensor, keep_fraction: f64) -> Tensor {
    let data = t.to_vec();
    let n = data.len();
    let k = (n as f64 * keep_fraction.clamp(0.0, 1.0)).ceil() as usize;
    let mut abs_vals: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, v)| (i, v.abs())).collect();
    abs_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0f64; n];
    for (idx, _) in abs_vals.iter().take(k) {
        result[*idx] = data[*idx];
    }
    Tensor::from_vec(result, t.shape().to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
