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

    #[test]
    fn test_vanilla_rnn_stress_001() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_002() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_003() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_004() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_005() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_006() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_007() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_008() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_009() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_010() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_011() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_012() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_013() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_014() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_015() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_016() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_017() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_018() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_019() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_020() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_021() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_022() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_023() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_024() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_025() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_026() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_027() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_028() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_029() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_030() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_031() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_032() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_033() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_034() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_035() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_036() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_037() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_038() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_039() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_040() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_041() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_042() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_043() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_044() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_045() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_046() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_047() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_048() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_049() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_050() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_051() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_052() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_053() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_054() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_055() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_056() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_057() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_058() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_059() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_060() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_061() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_062() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_063() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_064() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_065() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_066() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_067() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_068() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_069() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_070() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_071() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_072() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_073() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_074() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_075() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_076() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_077() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_078() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_079() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_080() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_081() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_082() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_083() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_084() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_085() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_086() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_087() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_088() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_089() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_090() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_091() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_092() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_093() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_094() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_095() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_096() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_097() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_098() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_099() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_100() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_101() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_102() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_103() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_104() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_105() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_106() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_107() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_108() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_109() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_110() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_111() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_112() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_113() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_114() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_115() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_116() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_117() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_118() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_119() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_120() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_121() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_122() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_123() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_124() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_125() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_126() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_127() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_128() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_129() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_130() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_131() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_132() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_133() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_134() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_135() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_136() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_137() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_138() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_139() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_140() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_141() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_142() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_143() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_144() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_145() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_146() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_147() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_148() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_149() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_150() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_151() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_152() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_153() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_154() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_155() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_156() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_157() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_158() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_159() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_160() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_161() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_162() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_163() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_164() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_165() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_166() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_167() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_168() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_169() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_170() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_171() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_172() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_173() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_174() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_175() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_176() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_177() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_178() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_179() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_180() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_181() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_182() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_183() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_184() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_185() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_186() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_187() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_188() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_189() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_190() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_191() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_192() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_193() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_194() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_195() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_196() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_197() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_198() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_199() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_200() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_201() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_202() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_203() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_204() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_205() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_206() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_207() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_208() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_209() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_210() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_211() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_212() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_213() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_214() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_215() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_216() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_217() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_218() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_219() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_220() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_221() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_222() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_223() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_224() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_225() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_226() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_227() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_228() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_229() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_230() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_231() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_232() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_233() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_234() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_235() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_236() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_237() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_238() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_239() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_240() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_241() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_242() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_243() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_244() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_245() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_246() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_247() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_248() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_249() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_250() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_251() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_252() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_253() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_254() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_255() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_256() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_257() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_258() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_259() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_260() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_261() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_262() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_263() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_264() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_265() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_266() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_267() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_268() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_269() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_270() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_271() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_272() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_273() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_274() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_275() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_276() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_277() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_278() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_279() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_280() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_281() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_282() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_283() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_284() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_285() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_286() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_287() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_288() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_289() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_290() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_291() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_292() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_293() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_294() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_295() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_296() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_297() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_298() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_299() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_300() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_301() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_302() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_303() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_304() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_305() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_306() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_307() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_308() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_309() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_310() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_311() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_312() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_313() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_314() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_315() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_316() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_317() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_318() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_319() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_320() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_321() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_322() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_323() {
        let cell = VanillaRnnCell::new(2, 4);
        let x = Tensor::from_slice(&[1.0, 2.0], vec![1, 2]);
        let s0 = cell.init_state(1);
        let (h1, s1) = cell.forward(&x, &s0).unwrap();
        assert_eq!(h1.shape(), &[1, 4]);
        assert_eq!(s1.hidden().shape(), &[1, 4]);
    }

    #[test]
    fn test_vanilla_rnn_stress_324() {
        let cell = VanillaRnnCell::new(2, 4);
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
}
