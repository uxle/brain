//! # Multi-Layer Stacked LSTM Sequence
//!
//! Stacked LSTM layers with inter-layer representations and initial state management.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::super::cells::{LstmCell, RnnCell};
use super::super::core::{CellState, RnnError, RnnResult, RnnState, SequenceOutput};
use super::RnnSequence;

/// Multi-layer Stacked LSTM Sequence Processor.
#[derive(Debug, Clone)]
pub struct LstmSeq {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub layers: Vec<LstmCell>,
}

impl LstmSeq {
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let num_layers = num_layers.max(1);
        let mut layers = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let in_d = if l == 0 { input_dim } else { hidden_dim };
            layers.push(LstmCell::new(in_d, hidden_dim));
        }

        Self {
            input_dim,
            hidden_dim,
            num_layers,
            layers,
        }
    }
}

impl RnnSequence for LstmSeq {
    fn forward(&self, input: &Tensor, init_state: Option<&RnnState>) -> RnnResult<SequenceOutput> {
        let s = input.shape();
        if s.len() != 3 {
            return Err(RnnError::ShapeMismatch { expected: vec![1, 1, self.input_dim], found: s.to_vec() });
        }

        let batch_size = s[0];
        let seq_len = s[1];
        let in_dim = s[2];

        if in_dim != self.input_dim {
            return Err(RnnError::DimensionMismatch { expected: self.input_dim, found: in_dim });
        }

        let mut current_states: Vec<CellState> = if let Some(st) = init_state {
            st.layer_states.clone()
        } else {
            self.init_state(batch_size).layer_states
        };

        let input_data = input.data();
        let mut layer_input = Vec::with_capacity(batch_size * seq_len * in_dim);
        layer_input.extend_from_slice(input_data);

        for l in 0..self.num_layers {
            let cur_in_dim = if l == 0 { self.input_dim } else { self.hidden_dim };
            let mut layer_output = vec![0.0; batch_size * seq_len * self.hidden_dim];

            for t in 0..seq_len {
                for b in 0..batch_size {
                    let start_idx = b * (seq_len * cur_in_dim) + t * cur_in_dim;
                    let x_t = Tensor::from_slice(&layer_input[start_idx..start_idx + cur_in_dim], vec![1, cur_in_dim]);

                    let (h_t, next_cell_state) = self.layers[l].forward(&x_t, &current_states[l])?;
                    current_states[l] = next_cell_state;

                    let h_data = h_t.data();
                    let out_idx = b * (seq_len * self.hidden_dim) + t * self.hidden_dim;
                    for d in 0..self.hidden_dim {
                        layer_output[out_idx + d] = h_data[d];
                    }
                }
            }

            layer_input = layer_output;
        }

        let out_tensor = Tensor::from_slice(&layer_input, vec![batch_size, seq_len, self.hidden_dim]);
        let final_rnn_state = RnnState::new(current_states);

        Ok(SequenceOutput::new(out_tensor, final_rnn_state))
    }

    fn init_state(&self, batch_size: usize) -> RnnState {
        let mut states = Vec::with_capacity(self.num_layers);
        for l in &self.layers {
            states.push(l.init_state(batch_size));
        }
        RnnState::new(states)
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
