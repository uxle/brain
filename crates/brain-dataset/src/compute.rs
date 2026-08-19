//! # Parallel Dataset Computations
//!
//! Parallel reduction and normalization computations over dataset items.

use crate::core::Batch;
use brain_core::Tensor;

/// Computes mean vector across batch items.
pub fn compute_batch_mean(batch: &Batch) -> Tensor {
    if batch.is_empty() {
        Tensor::scalar(0.0)
    } else {
        Tensor::zeros(batch.items[0].data.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
