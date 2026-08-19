//! # Recurrent Sequence Kernels & Packing Primitives
//!
//! Generic recurrent stepping, sequence padding, and packed sequence packing/unpacking.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

pub mod linear;
pub use linear::*;

use brain_core::Tensor;
use super::core::{RnnError, RnnResult};

/// Pads a list of variable length sequence tensors to uniform [batch, max_len, dim].
pub fn pad_sequence(sequences: &[Tensor], pad_value: f64) -> RnnResult<Tensor> {
    if sequences.is_empty() {
        return Err(RnnError::InvalidSequenceLength(0));
    }

    let batch_size = sequences.len();
    let mut max_len = 0;
    let mut feat_dim = 0;

    for seq in sequences {
        let s = seq.shape();
        if s.is_empty() {
            return Err(RnnError::ShapeMismatch { expected: vec![1, 1], found: s.to_vec() });
        }
        let len = s[0];
        let dim = if s.len() > 1 { s[1] } else { 1 };
        if max_len == 0 {
            feat_dim = dim;
        } else if dim != feat_dim {
            return Err(RnnError::DimensionMismatch { expected: feat_dim, found: dim });
        }
        if len > max_len {
            max_len = len;
        }
    }

    let mut padded = vec![pad_value; batch_size * max_len * feat_dim];
    for (b, seq) in sequences.iter().enumerate() {
        let seq_data = seq.data();
        let seq_len = seq.shape()[0];
        for t in 0..seq_len {
            for d in 0..feat_dim {
                padded[b * (max_len * feat_dim) + t * feat_dim + d] = seq_data[t * feat_dim + d];
            }
        }
    }

    Ok(Tensor::from_slice(&padded, vec![batch_size, max_len, feat_dim]))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
