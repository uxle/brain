//! # Distributed Tensor Reductions
//!
//! Elementwise tensor sum and average reductions across buffers.

use brain_core::Tensor;

/// Sums a collection of tensors elementwise.
pub fn reduce_sum(tensors: &[Tensor]) -> Tensor {
    if tensors.is_empty() {
        Tensor::scalar(0.0)
    } else {
        let mut acc = tensors[0].clone();
        for t in &tensors[1..] {
            acc = &acc + t;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
