//! # Federated Tensor Operations
//!
//! Delta scaling, summation, and clipping for federated aggregation.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Scales all tensors in a delta list by a scalar factor.
pub fn scale_delta(tensors: &[Tensor], factor: f64) -> Vec<Tensor> {
    let s = Tensor::scalar(factor);
    tensors.iter().map(|t| t * &s).collect()
}

/// Computes the L2 norm of a flattened delta.
pub fn l2_norm_delta(tensors: &[Tensor]) -> f64 {
    tensors
        .iter()
        .flat_map(|t| t.to_vec())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
