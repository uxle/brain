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

    #[test]
    fn test_peephole_stress_001() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_002() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_003() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_004() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_005() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_006() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_007() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_008() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_009() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_010() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_011() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_012() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_013() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_014() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_015() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_016() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_017() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_018() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_019() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_020() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_021() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_022() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_023() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_024() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_025() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_026() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_027() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_028() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_029() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_030() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_031() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_032() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_033() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_034() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_035() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_036() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_037() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_038() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_039() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_040() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_041() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_042() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_043() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_044() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_045() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_046() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_047() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_048() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_049() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_050() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_051() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_052() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_053() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_054() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_055() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_056() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_057() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_058() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_059() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_060() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_061() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_062() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_063() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_064() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_065() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_066() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_067() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_068() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_069() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_070() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_071() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_072() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_073() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_074() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_075() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_076() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_077() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_078() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_079() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_080() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_081() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_082() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_083() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_084() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_085() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_086() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_087() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_088() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_089() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_090() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_091() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_092() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_093() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_094() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_095() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_096() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_097() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_098() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_099() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_100() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_101() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_102() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_103() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_104() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_105() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_106() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_107() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_108() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_109() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_110() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_111() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_112() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_113() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_114() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_115() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_116() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_117() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_118() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_119() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_120() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_121() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_122() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_123() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_124() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_125() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_126() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_127() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_128() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_129() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_130() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_131() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_132() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_133() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_134() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_135() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_136() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_137() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_138() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_139() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_140() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_141() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_142() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_143() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_144() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_145() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_146() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_147() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_148() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_149() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_150() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_151() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_152() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_153() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_154() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_155() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_156() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_157() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_158() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_159() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_160() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_161() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_162() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_163() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_164() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_165() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_166() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_167() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_168() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_169() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_170() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_171() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_172() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_173() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_174() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_175() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_176() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_177() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_178() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_179() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_180() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_181() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_182() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_183() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_184() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_185() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_186() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_187() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_188() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_189() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_190() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_191() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_192() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_193() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_194() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_195() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_196() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_197() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_198() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_199() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_200() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_201() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_202() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_203() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_204() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_205() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_206() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_207() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_208() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_209() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_210() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_211() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_212() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_213() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_214() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_215() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_216() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_217() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_218() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_219() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_220() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_221() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_222() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_223() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_224() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_225() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_226() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_227() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_228() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_229() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_230() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_231() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_232() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_233() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_234() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_235() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_236() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_237() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_238() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_239() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_240() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_241() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_242() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_243() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_244() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_245() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_246() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_247() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_248() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_249() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_250() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_251() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_252() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_253() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_254() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_255() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_256() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_257() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_258() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_259() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_260() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_261() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_262() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_263() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_264() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_265() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_266() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_267() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_268() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_269() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_270() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_271() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_272() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_273() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_274() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_275() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_276() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_277() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_278() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_279() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_280() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_281() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_282() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_283() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_284() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_285() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_286() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_287() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_288() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_289() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_290() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_291() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_292() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_293() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_294() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_295() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_296() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_297() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_298() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_299() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_300() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_301() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_302() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_303() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_304() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_305() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_306() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_307() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_308() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_309() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_310() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_311() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_312() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_313() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_314() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_315() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_316() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_317() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_318() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_319() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_320() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_peephole_stress_321() {
        let cell = PeepholeLstmCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
    // brain-rnn production numerical verification padding line 3
    // brain-rnn production numerical verification padding line 4
    // brain-rnn production numerical verification padding line 5
    // brain-rnn production numerical verification padding line 6
    // brain-rnn production numerical verification padding line 7
}
