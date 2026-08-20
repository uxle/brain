//! # Sequence Processing Architecture & Abstractions
//!
//! Multi-layer sequence execution trait `RnnSequence` and sequence configuration parameters.
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

pub mod attention;
pub mod beam;
pub mod bidirectional;
pub mod gru_seq;
pub mod lstm_seq;
pub mod packed;
pub mod rnn_seq;
pub mod teacher;

pub use attention::SeqAttention;
pub use beam::{BeamConfig, BeamHypothesis, BeamSearch};
pub use bidirectional::{BidirectionalMerge, BidirectionalRnn};
pub use gru_seq::GruSeq;
pub use lstm_seq::LstmSeq;
pub use packed::PackedSequence;
pub use rnn_seq::VanillaRnnSeq;
pub use teacher::{TeacherForcer, TeacherSchedule};

use super::core::{RnnResult, RnnState, SequenceOutput};
use brain_core::Tensor;

/// Trait for multi-layer or bidirectional sequence-level RNN modules.
pub trait RnnSequence: Send + Sync {
    /// Unrolls sequence over all timesteps: inputs [batch, seq_len, in_dim].
    fn forward(&self, input: &Tensor, init_state: Option<&RnnState>) -> RnnResult<SequenceOutput>;

    /// Returns initial multi-layer zero state for given batch size.
    fn init_state(&self, batch_size: usize) -> RnnState;
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
