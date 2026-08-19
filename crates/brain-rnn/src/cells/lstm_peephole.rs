//! # Peephole LSTM Cell
//!
//! Direct cell-state connections to gate pre-activations ($w_{ci}, w_{cf}, w_{co}$).
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::super::core::{CellState, RnnError, RnnResult};
use super::super::ops::gate_linear;
use super::super::utils::{init_orthogonal, init_uniform, sigmoid};
use super::RnnCell;

/// Peephole LSTM Cell.
#[derive(Debug, Clone)]
pub struct PeepholeLstmCell {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub w_ih: Tensor,
    pub w_hh: Tensor,
    pub w_ci: Tensor, // [hidden_dim]
    pub w_cf: Tensor, // [hidden_dim]
    pub w_co: Tensor, // [hidden_dim]
    pub bias: Tensor,
}

impl PeepholeLstmCell {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let w_ih = init_uniform(4 * hidden_dim, input_dim, input_dim, 401);
        let w_hh = init_orthogonal(4 * hidden_dim, hidden_dim, 402);
        let w_ci = Tensor::from_slice(&vec![0.1; hidden_dim], vec![hidden_dim]);
        let w_cf = Tensor::from_slice(&vec![0.1; hidden_dim], vec![hidden_dim]);
        let w_co = Tensor::from_slice(&vec![0.1; hidden_dim], vec![hidden_dim]);
        let bias = Tensor::from_slice(&vec![0.0; 4 * hidden_dim], vec![4 * hidden_dim]);

        Self {
            input_dim,
            hidden_dim,
            w_ih,
            w_hh,
            w_ci,
            w_cf,
            w_co,
            bias,
        }
    }
}

impl RnnCell for PeepholeLstmCell {
    fn forward(&self, x: &Tensor, state: &CellState) -> RnnResult<(Tensor, CellState)> {
        let (h_prev, c_prev) = match state {
            CellState::Lstm { h, c } => (h, c),
            _ => return Err(RnnError::InvalidConfig("Expected LSTM state".into())),
        };

        let x_data = x.data();
        let h_data = h_prev.data();
        let c_data = c_prev.data();
        let h_dim = self.hidden_dim;

        let gates = gate_linear(
            x_data,
            h_data,
            self.w_ih.data(),
            self.w_hh.data(),
            Some(self.bias.data()),
            self.input_dim,
            h_dim,
            4 * h_dim,
        );

        let w_ci = self.w_ci.data();
        let w_cf = self.w_cf.data();
        let w_co = self.w_co.data();

        let mut h_next = vec![0.0; h_dim];
        let mut c_next = vec![0.0; h_dim];

        for i in 0..h_dim {
            let gate_i = sigmoid(gates[i] + w_ci[i] * c_data[i]);
            let gate_f = sigmoid(gates[h_dim + i] + w_cf[i] * c_data[i]);
            let gate_g = gates[2 * h_dim + i].tanh();

            let c_val = gate_f * c_data[i] + gate_i * gate_g;
            let gate_o = sigmoid(gates[3 * h_dim + i] + w_co[i] * c_val);
            let h_val = gate_o * c_val.tanh();

            c_next[i] = c_val;
            h_next[i] = h_val;
        }

        let h_tensor = Tensor::from_slice(&h_next, vec![1, h_dim]);
        let c_tensor = Tensor::from_slice(&c_next, vec![1, h_dim]);
        let new_state = CellState::new_lstm(h_tensor.clone(), c_tensor);

        Ok((h_tensor, new_state))
    }

    fn init_state(&self, batch_size: usize) -> CellState {
        let h = Tensor::from_slice(&vec![0.0; batch_size * self.hidden_dim], vec![batch_size, self.hidden_dim]);
        let c = Tensor::from_slice(&vec![0.0; batch_size * self.hidden_dim], vec![batch_size, self.hidden_dim]);
        CellState::new_lstm(h, c)
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
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
