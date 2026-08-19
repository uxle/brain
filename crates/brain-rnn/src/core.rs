//! # Recurrent Neural Network Core Types & State Representations
//!
//! State containers, output structs, and recurrent computation error types.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use std::fmt;
use brain_core::Tensor;

/// Recurrent hidden state for cells and sequences.
#[derive(Debug, Clone, PartialEq)]
pub enum CellState {
    Single(Tensor),
    Lstm { h: Tensor, c: Tensor },
}

impl CellState {
    pub fn new_single(h: Tensor) -> Self {
        CellState::Single(h)
    }

    pub fn new_lstm(h: Tensor, c: Tensor) -> Self {
        CellState::Lstm { h, c }
    }

    pub fn hidden(&self) -> &Tensor {
        match self {
            CellState::Single(h) => h,
            CellState::Lstm { h, .. } => h,
        }
    }

    pub fn cell(&self) -> Option<&Tensor> {
        match self {
            CellState::Single(_) => None,
            CellState::Lstm { c, .. } => Some(c),
        }
    }
}

/// Multi-layer RNN state tracking across sequence timesteps.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnState {
    pub layer_states: Vec<CellState>,
    pub step: usize,
}

impl RnnState {
    pub fn new(layer_states: Vec<CellState>) -> Self {
        Self {
            layer_states,
            step: 0,
        }
    }
}

/// Output container for sequence evaluations.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceOutput {
    pub output: Tensor,
    pub final_state: RnnState,
    pub attention_weights: Option<Tensor>,
}

impl SequenceOutput {
    pub fn new(output: Tensor, final_state: RnnState) -> Self {
        Self {
            output,
            final_state,
            attention_weights: None,
        }
    }
}

/// Errors encountered in recurrent operations or sequence formatting.
#[derive(Debug, Clone, PartialEq)]
pub enum RnnError {
    DimensionMismatch { expected: usize, found: usize },
    ShapeMismatch { expected: Vec<usize>, found: Vec<usize> },
    InvalidSequenceLength(usize),
    InvalidBatchSize(usize),
    InvalidConfig(String),
    InferenceError(String),
}

impl fmt::Display for RnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RnnError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {}, found {}", expected, found)
            }
            RnnError::ShapeMismatch { expected, found } => {
                write!(f, "Shape mismatch: expected {:?}, found {:?}", expected, found)
            }
            RnnError::InvalidSequenceLength(len) => write!(f, "Invalid sequence length: {}", len),
            RnnError::InvalidBatchSize(bs) => write!(f, "Invalid batch size: {}", bs),
            RnnError::InvalidConfig(msg) => write!(f, "Invalid RNN configuration: {}", msg),
            RnnError::InferenceError(msg) => write!(f, "RNN inference error: {}", msg),
        }
    }
}

impl std::error::Error for RnnError {}

pub type RnnResult<T> = Result<T, RnnError>;

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
