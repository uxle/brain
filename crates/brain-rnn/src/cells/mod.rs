//! # Recurrent Cell Trait & Family Abstractions
//!
//! Standard `RnnCell` interface, state transitions, and parameter layout conventions.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

pub mod lstm;
pub mod gru;
pub mod rnn;
pub mod lstm_peephole;
pub mod attention_cell;
pub mod normalized;

pub use lstm::LstmCell;
pub use gru::GruCell;
pub use rnn::VanillaRnnCell;
pub use lstm_peephole::PeepholeLstmCell;
pub use attention_cell::AttentionCell;
pub use normalized::NormLstmCell;

use brain_core::Tensor;
use super::core::{CellState, RnnResult};

/// Universal trait for individual recurrent step cells.
pub trait RnnCell: Send + Sync {
    /// Advances cell by a single step: $(x_t, s_{t-1}) \mapsto (h_t, s_t)$.
    fn forward(&self, x: &Tensor, state: &CellState) -> RnnResult<(Tensor, CellState)>;

    /// Returns initial zero state matching batch size of input.
    fn init_state(&self, batch_size: usize) -> CellState;

    /// Input feature dimension.
    fn input_dim(&self) -> usize;

    /// Hidden state dimension.
    fn hidden_dim(&self) -> usize;
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
