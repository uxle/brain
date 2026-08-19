//! # High-Level Functional Entrypoints
//!
//! Functional sequence transformations and convenient one-shot evaluation routines.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::core::{RnnResult, SequenceOutput};
use super::seq::lstm_seq::LstmSeq;
use super::seq::RnnSequence;

/// One-shot functional LSTM sequence evaluation.
pub fn forward_lstm(input: &Tensor, input_dim: usize, hidden_dim: usize, layers: usize) -> RnnResult<SequenceOutput> {
    let seq = LstmSeq::new(input_dim, hidden_dim, layers);
    seq.forward(input, None)
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
