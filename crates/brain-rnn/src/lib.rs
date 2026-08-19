//! # Brain Recurrent Neural Network Framework (`brain-rnn`)
//!
//! Production-grade recurrent architectures: LSTM, GRU, Vanilla RNN, Attention, Bidirectional, and Packed Sequences.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

pub mod core;
pub mod config;
pub mod utils;
pub mod ops;
pub mod cells;
pub mod seq;
pub mod init_rnn;
pub mod reg_ops;
pub mod process;
pub mod backward_ops;
pub mod builder;
pub mod helper;
pub mod r#impl;

pub use core::{CellState, RnnError, RnnResult, RnnState, SequenceOutput};
pub use config::{CellConfig, CellKind, RnnConfig};
pub use cells::{AttentionCell, GruCell, LstmCell, NormLstmCell, PeepholeLstmCell, RnnCell, VanillaRnnCell};
pub use seq::{BidirectionalMerge, BidirectionalRnn, GruSeq, LstmSeq, PackedSequence, RnnSequence, SeqAttention, VanillaRnnSeq};
pub use builder::RnnBuilder;
pub use helper::create_padding_mask;
pub use r#impl::forward_lstm;

/// Semantic version of the `brain-rnn` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key cell and sequence abstractions.
pub mod prelude {
    pub use super::core::{CellState, RnnError, RnnResult, RnnState, SequenceOutput};
    pub use super::config::{CellConfig, CellKind, RnnConfig};
    pub use super::cells::{GruCell, LstmCell, PeepholeLstmCell, RnnCell, VanillaRnnCell};
    pub use super::seq::{BidirectionalMerge, BidirectionalRnn, GruSeq, LstmSeq, PackedSequence, RnnSequence, VanillaRnnSeq};
    pub use super::builder::RnnBuilder;
    pub use super::r#impl::forward_lstm;
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
