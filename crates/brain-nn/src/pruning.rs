//! # Neural Network Weight Pruning
//!
//! Unstructured magnitude pruning, structured channel pruning, and binary pruning masks.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Binary pruning mask applied elementwise to parameter tensors.
#[derive(Debug, Clone)]
pub struct PruningMask {
    pub mask: Tensor,
}

impl PruningMask {
    pub fn from_magnitude(weight: &Tensor, sparsity: f64) -> Self {
        let data = weight.to_vec();
        let mut abs_vals: Vec<f64> = data.iter().map(|&x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let k = ((abs_vals.len() as f64) * sparsity).floor() as usize;
        let threshold = if k < abs_vals.len() {
            abs_vals[k]
        } else {
            f64::INFINITY
        };

        let mask_data: Vec<f64> = data
            .iter()
            .map(|&x| if x.abs() >= threshold { 1.0 } else { 0.0 })
            .collect();
        Self {
            mask: Tensor::from_vec(mask_data, weight.shape().to_vec()),
        }
    }

    pub fn apply(&self, weight: &Tensor) -> Tensor {
        weight * &self.mask
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
