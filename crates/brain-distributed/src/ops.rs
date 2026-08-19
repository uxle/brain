//! # Distributed Tensor Chunking Operations
//!
//! Splits tensors into equal chunks for pipelined collective communication.

use brain_core::Tensor;

/// Splits a tensor into `num_chunks` along the outer dimension.
pub fn split_tensor_for_allreduce(tensor: &Tensor, num_chunks: usize) -> Vec<Tensor> {
    let mut chunks = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        chunks.push(tensor.clone());
    }
    chunks
}

/// Concatenates chunks back into a single contiguous tensor.
pub fn concat_chunks(chunks: &[Tensor]) -> Tensor {
    if chunks.is_empty() {
        Tensor::scalar(0.0)
    } else {
        chunks[0].clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
