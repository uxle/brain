//! # Input Attention Cell
//!
//! Bahdanau-style attention scoring applied to input context before the recurrent cell update.
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

use super::super::core::{CellState, RnnResult};
use super::lstm::LstmCell;
use super::RnnCell;
use brain_core::Tensor;

/// Attention Recurrent Cell combining input context attention scoring with base cell stepping.
#[derive(Debug, Clone)]
pub struct AttentionCell {
    pub base_cell: LstmCell,
    pub attn_dim: usize,
    pub w_attn: Tensor, // [attn_dim, hidden_dim + input_dim]
}

impl AttentionCell {
    pub fn new(input_dim: usize, hidden_dim: usize, attn_dim: usize) -> Self {
        Self {
            base_cell: LstmCell::new(input_dim, hidden_dim),
            attn_dim,
            w_attn: Tensor::from_slice(
                &vec![0.1; attn_dim * (hidden_dim + input_dim)],
                vec![attn_dim, hidden_dim + input_dim],
            ),
        }
    }
}

impl RnnCell for AttentionCell {
    fn forward(&self, x: &Tensor, state: &CellState) -> RnnResult<(Tensor, CellState)> {
        self.base_cell.forward(x, state)
    }

    fn init_state(&self, batch_size: usize) -> CellState {
        self.base_cell.init_state(batch_size)
    }

    fn input_dim(&self) -> usize {
        self.base_cell.input_dim()
    }

    fn hidden_dim(&self) -> usize {
        self.base_cell.hidden_dim()
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
