//! # Framework Interoperability Adapters
//!
//! Bridges `brain-data` pipeline abstractions with `brain-core::Tensor` data structures.

use crate::core::Sample;
use brain_core::Tensor;

/// Converts a tensor slice into a sequence of samples.
pub fn tensor_to_samples(tensor: &Tensor) -> Vec<Sample> {
    let num_samples = tensor.shape()[0];
    (0..num_samples)
        .map(|i| Sample::new(i, tensor.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
