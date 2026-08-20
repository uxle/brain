//! # Packed Sequence Containers
//!
//! Compact memory layout for variable-length padded batch sequences.
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

/// Packed variable-length sequence container.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedSequence {
    pub data: Tensor,
    pub batch_sizes: Vec<usize>,
    pub sorted_indices: Vec<usize>,
}

impl PackedSequence {
    pub fn new(data: Tensor, batch_sizes: Vec<usize>, sorted_indices: Vec<usize>) -> Self {
        Self {
            data,
            batch_sizes,
            sorted_indices,
        }
    }

    pub fn num_timesteps(&self) -> usize {
        self.batch_sizes.len()
    }
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
