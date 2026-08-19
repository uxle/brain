//! # Sequence Processing & Online Streaming
//!
//! Chunked sequence evaluation and real-time streaming stateful updates.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::core::{CellState, RnnResult};
use super::cells::{LstmCell, RnnCell};

/// Online streaming processor maintaining state across arbitrary timestep chunks.
#[derive(Debug, Clone)]
pub struct OnlineRnnStreamer {
    pub cell: LstmCell,
    pub current_state: CellState,
}

impl OnlineRnnStreamer {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let cell = LstmCell::new(input_dim, hidden_dim);
        let current_state = cell.init_state(1);
        Self { cell, current_state }
    }

    /// Feeds next input token and updates persistent internal recurrent state.
    pub fn feed_step(&mut self, x: &Tensor) -> RnnResult<Tensor> {
        let (h, next_state) = self.cell.forward(x, &self.current_state)?;
        self.current_state = next_state;
        Ok(h)
    }

    pub fn reset(&mut self) {
        self.current_state = self.cell.init_state(1);
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
