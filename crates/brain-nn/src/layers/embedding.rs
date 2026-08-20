//! # Lookup Embedding Layers
//!
//! Discrete token and index embeddings with optional padding index masking and positional encodings.
//!
//! Migrated in Phase 0. Note this layer intentionally does NOT implement the
//! generic `Module` trait, before or after this migration: `Module::forward`
//! takes a `&Value` (a differentiable tensor input), but `Embedding`'s
//! natural input is a slice of discrete token indices (`&[usize]`), which
//! are not differentiable -- there is no meaningful gradient with respect to
//! "which row index was looked up". This mirrors how PyTorch's own
//! `nn.Embedding` treats its index input as non-differentiable even though
//! it's wrapped in a `LongTensor`. Forcing this layer to implement `Module`
//! would mean pretending indices are a `Value` for no real benefit, so it
//! keeps its own `forward_indices` method instead -- this was already true
//! before Phase 0 and is preserved deliberately, not an oversight.
//!
//! Known pre-existing gap, NOT introduced by this migration, flagged for
//! visibility: `forward_indices` (both before and after this change) silently
//! substitutes a zero vector for any out-of-range index instead of erroring.
//! That's a separate correctness question worth its own follow-up (likely
//! Stage A/D-style: should this panic, return a `Result`, or is silent
//! zero-fill intentional for a specific use case like padding tokens?) --
//! left untouched here to keep this migration scoped to the Tensor->Value
//! change specifically.
#![allow(missing_docs)]

use crate::init::normal_init;
use brain_autograd::Value;
use brain_core::Tensor;

/// Token lookup embedding table: [num_embeddings, embedding_dim].
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Value,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight_tensor = normal_init(&[num_embeddings, embedding_dim], 0.0, 1.0);
        Self {
            weight: Value::new(weight_tensor, true),
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        }
    }

    /// Looks up `indices` against the embedding table and returns a
    /// tape-tracked `Value` of shape `[indices.len(), embedding_dim]`.
    /// Unlike before Phase 0, gradients now flow back into `self.weight`
    /// correctly on repeated indices -- `Value::embedding`'s backward
    /// accumulates (sums) gradient contributions for duplicate indices in
    /// the same batch, rather than overwriting, matching the standard
    /// embedding-gradient semantics verified elsewhere in the gap audit.
    pub fn forward_indices(&self, indices: &[usize]) -> Value {
        let output_shape = vec![indices.len(), self.embedding_dim];
        self.weight.embedding(indices, output_shape)
    }

    /// Parameters for this layer -- exposed as an inherent method since
    /// `Embedding` doesn't implement `Module` (see module doc comment).
    /// A `ModuleList`/`ModuleDict` cannot currently hold an `Embedding`
    /// alongside `Module`-implementing layers because of this; that's a
    /// pre-existing limitation of the trait design, not something this
    /// migration changes. Worth a follow-up: either give `Module` an
    /// optional index-input hook, or a separate small trait for
    /// index-input layers that containers can also aggregate over.
    pub fn parameters(&self) -> Vec<Value> {
        vec![self.weight.clone()]
    }
}

/// Generates sinusoidal positional encoding table of shape [seq_len, embedding_dim].
/// Unchanged -- this is a fixed, non-learned table (no parameters, nothing
/// to differentiate), so it stays a plain `Tensor`-producing function.
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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_embedding_lookup_shape_via_tape() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 3, 5]);
        assert_eq!(out.shape(), &[3, 4]);
    }

    /// The gradient-accumulation-on-duplicate-index test this layer never
    /// had a real backward path to check before Phase 0. Confirms the
    /// weight gradient correctly SUMS contributions when the same index
    /// appears twice in one batch, not overwrites.
    #[test]
    fn test_embedding_duplicate_index_gradient_accumulates_via_tape() {
        let emb = Embedding::new(5, 2);
        // Index 2 appears twice -- its weight row's gradient should be the
        // sum of both positions' upstream gradient, not just the last one.
        let out = emb.forward_indices(&[2, 2, 0]);
        let loss = out.sum();
        loss.backward().unwrap();

        let grad = emb.weight.grad().unwrap().to_vec();
        // Row 2 (embedding_dim=2, so indices 4,5) received gradient from
        // TWO lookups summing to 1.0 each element (d(sum)/d(each output
        // element) = 1.0), row 0 (indices 0,1) from ONE lookup.
        assert_eq!(
            grad[4], 2.0,
            "row 2, col 0 should accumulate from both lookups"
        );
        assert_eq!(
            grad[5], 2.0,
            "row 2, col 1 should accumulate from both lookups"
        );
        assert_eq!(grad[0], 1.0, "row 0, col 0 should reflect a single lookup");
        assert_eq!(grad[1], 1.0, "row 0, col 1 should reflect a single lookup");
        // Row 1, 3, 4 were never looked up -- should have zero (or no)
        // contribution.
        assert_eq!(grad[2], 0.0);
        assert_eq!(grad[3], 0.0);
    }
}
