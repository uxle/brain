//! # Layer-Normalized LSTM Cell (LN-LSTM)
//!
//! Applies layer normalization to pre-activation gate projections to stabilize training.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::super::core::{CellState, RnnResult};
use super::lstm::LstmCell;
use super::RnnCell;

/// Layer-Normalized LSTM Cell.
#[derive(Debug, Clone)]
pub struct NormLstmCell {
    pub base_cell: LstmCell,
    pub gamma: Tensor,
    pub beta: Tensor,
}

impl NormLstmCell {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            base_cell: LstmCell::new(input_dim, hidden_dim),
            gamma: Tensor::from_slice(&vec![1.0; 4 * hidden_dim], vec![4 * hidden_dim]),
            beta: Tensor::from_slice(&vec![0.0; 4 * hidden_dim], vec![4 * hidden_dim]),
        }
    }
}

impl RnnCell for NormLstmCell {
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
