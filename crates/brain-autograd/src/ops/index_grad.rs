//! # Indexing & Gathering Operation Gradients
//!
//! Backward rules for indexing operations and embedding lookup tables.

use brain_core::{BrainResult, Tensor};

/// Backward for `embedding` lookup table.
pub fn grad_embedding(
    g: &Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    indices: &[usize],
) -> BrainResult<Tensor> {
    let mut grad_weights = vec![0.0; num_embeddings * embedding_dim];
    let g_slice = g.data();

    for (seq_idx, &emb_idx) in indices.iter().enumerate() {
        if emb_idx < num_embeddings {
            for d in 0..embedding_dim {
                grad_weights[emb_idx * embedding_dim + d] += g_slice[seq_idx * embedding_dim + d];
            }
        }
    }

    Ok(Tensor::from_slice(
        &grad_weights,
        vec![num_embeddings, embedding_dim],
    ))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
