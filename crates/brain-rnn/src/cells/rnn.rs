//! # Vanilla Elman RNN Cell
//!
//! Classic single-gate recurrent cell with Tanh non-linearity.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::super::core::{CellState, RnnError, RnnResult};
use super::super::ops::gate_linear;
use super::super::utils::{init_orthogonal, init_uniform};
use super::RnnCell;

/// Vanilla Elman RNN Cell: $h_t = \tanh(W x_t + U h_{t-1} + b)$.
#[derive(Debug, Clone)]
pub struct VanillaRnnCell {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub w_ih: Tensor,
    pub w_hh: Tensor,
    pub bias: Tensor,
}

impl VanillaRnnCell {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let w_ih = init_uniform(hidden_dim, input_dim, input_dim, 301);
        let w_hh = init_orthogonal(hidden_dim, hidden_dim, 302);
        let bias = Tensor::from_slice(&vec![0.0; hidden_dim], vec![hidden_dim]);

        Self {
            input_dim,
            hidden_dim,
            w_ih,
            w_hh,
            bias,
        }
    }
}

impl RnnCell for VanillaRnnCell {
    fn forward(&self, x: &Tensor, state: &CellState) -> RnnResult<(Tensor, CellState)> {
        let h_prev = match state {
            CellState::Single(h) => h,
            _ => return Err(RnnError::InvalidConfig("Expected Single RNN state".into())),
        };

        let x_data = x.data();
        let h_data = h_prev.data();
        let h_dim = self.hidden_dim;

        let pre = gate_linear(
            x_data,
            h_data,
            self.w_ih.data(),
            self.w_hh.data(),
            Some(self.bias.data()),
            self.input_dim,
            h_dim,
            h_dim,
        );

        let mut h_next = vec![0.0; h_dim];
        for i in 0..h_dim {
            h_next[i] = pre[i].tanh();
        }

        let h_tensor = Tensor::from_slice(&h_next, vec![1, h_dim]);
        let new_state = CellState::new_single(h_tensor.clone());

        Ok((h_tensor, new_state))
    }

    fn init_state(&self, batch_size: usize) -> CellState {
        let h = Tensor::from_slice(&vec![0.0; batch_size * self.hidden_dim], vec![batch_size, self.hidden_dim]);
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
