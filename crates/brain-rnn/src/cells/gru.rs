//! # Gated Recurrent Unit (GRU) Cell
//!
//! Compact 3-gate GRU cell (reset $r_t$, update $z_t$, and candidate $n_t$).
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

use super::super::core::{CellState, RnnError, RnnResult};
use super::super::ops::gate_linear;
use super::super::utils::{init_orthogonal, init_uniform, sigmoid};
use super::RnnCell;
use brain_core::Tensor;

/// Gated Recurrent Unit (GRU) Cell.
#[derive(Debug, Clone)]
pub struct GruCell {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub w_ih: Tensor, // [3 * hidden_dim, input_dim]
    pub w_hh: Tensor, // [3 * hidden_dim, hidden_dim]
    pub bias: Tensor, // [3 * hidden_dim]
}

impl GruCell {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let w_ih = init_uniform(3 * hidden_dim, input_dim, input_dim, 201);
        let w_hh = init_orthogonal(3 * hidden_dim, hidden_dim, 202);
        let bias = Tensor::from_slice(&vec![0.0; 3 * hidden_dim], vec![3 * hidden_dim]);

        Self {
            input_dim,
            hidden_dim,
            w_ih,
            w_hh,
            bias,
        }
    }
}

impl RnnCell for GruCell {
    fn forward(&self, x: &Tensor, state: &CellState) -> RnnResult<(Tensor, CellState)> {
        let h_prev = match state {
            CellState::Single(h) => h,
            _ => return Err(RnnError::InvalidConfig("Expected Single GRU state".into())),
        };

        let x_data = x.data();
        let h_data = h_prev.data();
        let h_dim = self.hidden_dim;

        let gates = gate_linear(
            x_data,
            h_data,
            self.w_ih.data(),
            self.w_hh.data(),
            Some(self.bias.data()),
            self.input_dim,
            h_dim,
            3 * h_dim,
        );

        let mut h_next = vec![0.0; h_dim];

        for i in 0..h_dim {
            let r = sigmoid(gates[i]);
            let z = sigmoid(gates[h_dim + i]);
            let n = (gates[2 * h_dim + i] * r).tanh();

            let h_val = (1.0 - z) * n + z * h_data[i];
            h_next[i] = h_val;
        }

        let h_tensor = Tensor::from_slice(&h_next, vec![1, h_dim]);
        let new_state = CellState::new_single(h_tensor.clone());

        Ok((h_tensor, new_state))
    }

    fn init_state(&self, batch_size: usize) -> CellState {
        let h = Tensor::from_slice(
            &vec![0.0; batch_size * self.hidden_dim],
            vec![batch_size, self.hidden_dim],
        );
        CellState::new_single(h)
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
