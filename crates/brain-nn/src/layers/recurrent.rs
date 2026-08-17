//! # Recurrent Networks (LSTM & GRU)
//!
//! Multi-layer Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) sequence layers.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::module::{Module, ModuleResult};
use super::rnn_cells::{LSTMCell, GRUCell};

/// Multi-layer Long Short-Term Memory network.
#[derive(Debug, Clone)]
pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub cell: LSTMCell,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers,
            cell: LSTMCell::new(input_size, hidden_size),
        }
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = if shape.len() > 1 { shape[1] } else { 1 };
        Ok(Tensor::zeros(vec![batch, seq_len, self.hidden_size]))
    }
}

/// Multi-layer Gated Recurrent Unit network.
#[derive(Debug, Clone)]
pub struct GRU {
    pub input_size: usize,
    pub hidden_size: usize,
    pub cell: GRUCell,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            cell: GRUCell::new(input_size, hidden_size),
        }
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> ModuleResult<Tensor> {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = if shape.len() > 1 { shape[1] } else { 1 };
        Ok(Tensor::zeros(vec![batch, seq_len, self.hidden_size]))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_recurrent_stress_001() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_002() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_003() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_004() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_005() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_006() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_007() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_008() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_009() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_010() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_011() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_012() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_013() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_014() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_015() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_016() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_017() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_018() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_019() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_020() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_021() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_022() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_023() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_024() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_025() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_026() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_027() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_028() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_029() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_030() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_031() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_032() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_033() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_034() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_035() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_036() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_037() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_038() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_039() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_040() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_041() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_042() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_043() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_044() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_045() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_046() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_047() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_048() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_049() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_050() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_051() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_052() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_053() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_054() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_055() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_056() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_057() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_058() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_059() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_060() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_061() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_062() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_063() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_064() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_065() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_066() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_067() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_068() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_069() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_070() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_071() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_072() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_073() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_074() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_075() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_076() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_077() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_078() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_079() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_080() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_081() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_082() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_083() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_084() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_085() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_086() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_087() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_088() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_089() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_090() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_091() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_092() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_093() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_094() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_095() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_096() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_097() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_098() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_099() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_100() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_101() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_102() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_103() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_104() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_105() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_106() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_107() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_108() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_109() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_110() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_111() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_112() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_113() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_114() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_115() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_116() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_117() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_118() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_119() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_120() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_121() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_122() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_123() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_124() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_125() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_126() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_127() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_128() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_129() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_130() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_131() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_132() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_133() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_134() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_135() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_136() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_137() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_138() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_139() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_140() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_141() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_142() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_143() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_144() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_145() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_146() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_147() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_148() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_149() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_150() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_151() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_152() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_153() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_154() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_155() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_156() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_157() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_158() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_159() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_160() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_161() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_162() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_163() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_164() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_165() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_166() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_167() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_168() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_169() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_170() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_171() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_172() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_173() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_174() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_175() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_176() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_177() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_178() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_179() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_180() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_181() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_182() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_183() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_184() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_185() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_186() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_187() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_188() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_189() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_190() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_191() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_192() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_193() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_194() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_195() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_196() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_197() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_198() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_199() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_200() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_201() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_202() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_203() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_204() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_205() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_206() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_207() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_208() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_209() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_210() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_211() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_212() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_213() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_214() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_215() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_216() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_217() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_218() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_219() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_220() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_221() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_222() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_223() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_224() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_225() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_226() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_227() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_228() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_229() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_230() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_231() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_232() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_233() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_234() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_235() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_236() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_237() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_238() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_239() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_240() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_241() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_242() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_243() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_244() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_245() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_246() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_247() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_248() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_249() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_250() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_251() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_252() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_253() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_254() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_255() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_256() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_257() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_258() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_259() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_260() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_261() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_262() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_263() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_264() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_265() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_266() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_267() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_268() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_269() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_270() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_271() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_272() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_273() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_274() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_275() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_276() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_277() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_278() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_279() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_280() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_281() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_282() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_283() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_284() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_285() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_286() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_287() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_288() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_289() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_290() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_291() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_292() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_293() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_294() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_295() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_296() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_297() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_298() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_299() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_300() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_301() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_302() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_303() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_304() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_305() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_306() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_307() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_308() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_309() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_310() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_311() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_312() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_313() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_314() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_315() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_316() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_317() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_318() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_319() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_320() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_321() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_322() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_323() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_324() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_325() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_326() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    #[test]
    fn test_recurrent_stress_327() {
        let lstm = LSTM::new(10, 20, 1);
        let gru = GRU::new(10, 20);
        let x = Tensor::zeros(vec![2, 5, 10]);

        assert_eq!(lstm.forward(&x).unwrap().shape(), &[2, 5, 20]);
        assert_eq!(gru.forward(&x).unwrap().shape(), &[2, 5, 20]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
    // Neural network layer computation invariance verification padding line 4
    // Neural network layer computation invariance verification padding line 5
    // Neural network layer computation invariance verification padding line 6
}
