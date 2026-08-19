//! # Sequence Masking & Length Helpers
//!
//! Boolean sequence padding masks and length conversion utilities.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;

/// Generates boolean attention / padding mask: shape $[\text{batch}, \text{max\_len}]$.
pub fn create_padding_mask(lengths: &[usize], max_len: usize) -> Tensor {
    let batch_size = lengths.len();
    let mut mask = vec![0.0; batch_size * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        for t in 0..len.min(max_len) {
            mask[b * max_len + t] = 1.0;
        }
    }
    Tensor::from_slice(&mask, vec![batch_size, max_len])
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
