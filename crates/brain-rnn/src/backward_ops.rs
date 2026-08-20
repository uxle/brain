//! # Truncated Backpropagation Through Time (TBPTT) Utilities
//!
//! Sequence chunking and gradient truncation windowing helpers.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown,
    clippy::module_inception,
    clippy::manual_memcpy
)]

use brain_core::Tensor;

/// Splits a long sequence into truncated BPTT chunks of length `chunk_size`.
pub fn truncate_steps(sequence: &Tensor, chunk_size: usize) -> Vec<Tensor> {
    let s = sequence.shape();
    let batch_size = s[0];
    let seq_len = s[1];
    let dim = s[2];
    let d = sequence.data();

    let num_chunks = (seq_len + chunk_size - 1) / chunk_size.max(1);
    let mut chunks = Vec::with_capacity(num_chunks);

    for c in 0..num_chunks {
        let start_t = c * chunk_size;
        let cur_len = (seq_len - start_t).min(chunk_size);
        let mut chunk_data = vec![0.0; batch_size * cur_len * dim];

        for b in 0..batch_size {
            for t in 0..cur_len {
                let src_idx = b * (seq_len * dim) + (start_t + t) * dim;
                let dst_idx = b * (cur_len * dim) + t * dim;
                for i in 0..dim {
                    chunk_data[dst_idx + i] = d[src_idx + i];
                }
            }
        }

        chunks.push(Tensor::from_slice(
            &chunk_data,
            vec![batch_size, cur_len, dim],
        ));
    }

    chunks
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision
    )]
    use super::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::cells::*;
    use crate::config::*;
    use crate::core::*;
    use crate::helper::*;
    use crate::init_rnn::*;
    use crate::ops::*;
    use crate::process::*;
    use crate::reg_ops::*;
    use crate::seq::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
