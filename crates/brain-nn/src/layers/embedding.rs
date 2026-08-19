//! # Lookup Embedding Layers
//!
//! Discrete token and index embeddings with optional padding index masking and positional encodings.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::init::normal_init;

/// Token lookup embedding table: [num_embeddings, embedding_dim].
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = normal_init(&[num_embeddings, embedding_dim], 0.0, 1.0);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        }
    }

    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        let n = indices.len();
        let mut data = Vec::with_capacity(n * self.embedding_dim);
        let w_data = self.weight.to_vec();

        for &idx in indices {
            if idx < self.num_embeddings {
                let slice = &w_data[idx * self.embedding_dim..(idx + 1) * self.embedding_dim];
                data.extend_from_slice(slice);
            } else {
                data.extend(vec![0.0; self.embedding_dim]);
            }
        }

        Tensor::from_vec(data, vec![n, self.embedding_dim])
    }
}

/// Generates sinusoidal positional encoding table of shape [seq_len, embedding_dim].
pub fn sinusoidal_positional_encoding(seq_len: usize, embedding_dim: usize) -> Tensor {
    let mut data = vec![0.0f64; seq_len * embedding_dim];
    for pos in 0..seq_len {
        for i in 0..embedding_dim / 2 {
            let div_term = (10000.0_f64).powf((2 * i) as f64 / embedding_dim as f64);
            let angle = pos as f64 / div_term;
            data[pos * embedding_dim + 2 * i] = angle.sin();
            data[pos * embedding_dim + 2 * i + 1] = angle.cos();
        }
    }
    Tensor::from_vec(data, vec![seq_len, embedding_dim])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
