//! # Gradient Bucketing & Overlapping
//!
//! Groups small gradients into fixed-size contiguous buckets to maximize network bandwidth.

use brain_core::Tensor;

/// Gradient bucket for coalesced communication.
pub struct GradBucket {
    pub max_bytes: usize,
    pub tensors: Vec<Tensor>,
}

impl GradBucket {
    /// Creates a new `GradBucket`.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            tensors: Vec::new(),
        }
    }

    /// Adds a tensor to the bucket.
    pub fn push(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
