//! # Recurrent Networks (LSTM & GRU)
//!
//! Multi-layer Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) sequence layers.
#![allow(missing_docs)]

use super::rnn_cells::{GRUCell, LSTMCell};
use crate::module::{Module, ModuleError, ModuleResult};
use brain_core::Tensor;

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

use brain_autograd::Value;

impl Module for LSTM {
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() < 3 || shape[2] != self.input_size {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![shape.first().copied().unwrap_or(1), 1, self.input_size],
                got: shape.to_vec(),
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let in_data = input.to_vec();

        let mut h = Tensor::zeros(vec![batch, self.hidden_size]);
        let mut c = Tensor::zeros(vec![batch, self.hidden_size]);
        let mut out_all = Vec::with_capacity(batch * seq_len * self.hidden_size);

        // We collect outputs per timestep: out shape [batch, seq_len, hidden_size]
        let mut step_outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let mut x_t_vec = Vec::with_capacity(batch * self.input_size);
            for bi in 0..batch {
                let start = (bi * seq_len + t) * self.input_size;
                x_t_vec.extend_from_slice(&in_data[start..start + self.input_size]);
            }
            let x_t = Tensor::from_vec(x_t_vec, vec![batch, self.input_size]);
            let (next_h, next_c) = self.cell.forward_step(&x_t, &h, &c);
            h = next_h;
            c = next_c;
            step_outputs.push(h.to_vec());
        }

        for bi in 0..batch {
            for t in 0..seq_len {
                let h_slice = &step_outputs[t][bi * self.hidden_size..(bi + 1) * self.hidden_size];
                out_all.extend_from_slice(h_slice);
            }
        }

        let t_out = Tensor::from_vec(out_all, vec![batch, seq_len, self.hidden_size]);
        Ok(Value::new(t_out, input.requires_grad()))
    }

    fn parameters(&self) -> Vec<Value> {
        self.cell
            .parameters()
            .into_iter()
            .map(|t| Value::new(t, true))
            .collect()
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
    fn forward(&self, input: &Value) -> ModuleResult<Value> {
        let shape = input.shape();
        if shape.len() < 3 || shape[2] != self.input_size {
            return Err(ModuleError::ShapeMismatch {
                expected: vec![shape.first().copied().unwrap_or(1), 1, self.input_size],
                got: shape.to_vec(),
            });
        }
        let batch = shape[0];
        let seq_len = shape[1];
        let in_data = input.to_vec();

        let mut h = Tensor::zeros(vec![batch, self.hidden_size]);
        let mut step_outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let mut x_t_vec = Vec::with_capacity(batch * self.input_size);
            for bi in 0..batch {
                let start = (bi * seq_len + t) * self.input_size;
                x_t_vec.extend_from_slice(&in_data[start..start + self.input_size]);
            }
            let x_t = Tensor::from_vec(x_t_vec, vec![batch, self.input_size]);
            h = self.cell.forward_step(&x_t, &h);
            step_outputs.push(h.to_vec());
        }

        let mut out_all = Vec::with_capacity(batch * seq_len * self.hidden_size);
        for bi in 0..batch {
            for t in 0..seq_len {
                let h_slice = &step_outputs[t][bi * self.hidden_size..(bi + 1) * self.hidden_size];
                out_all.extend_from_slice(h_slice);
            }
        }

        let t_out = Tensor::from_vec(out_all, vec![batch, seq_len, self.hidden_size]);
        Ok(Value::new(t_out, input.requires_grad()))
    }

    fn parameters(&self) -> Vec<Value> {
        self.cell
            .parameters()
            .into_iter()
            .map(|t| Value::new(t, true))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_lstm_forward_shape_and_values() {
        let lstm = LSTM::new(4, 8, 1);
        let x = Value::new(Tensor::from_vec(vec![1.0; 2 * 5 * 4], vec![2, 5, 4]), false);
        let out = lstm.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 5, 8]);
        assert!(out.to_vec().iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_gru_forward_shape_and_values() {
        let gru = GRU::new(4, 8);
        let x = Value::new(Tensor::from_vec(vec![1.0; 2 * 5 * 4], vec![2, 5, 4]), false);
        let out = gru.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 5, 8]);
        assert!(out.to_vec().iter().any(|&v| v != 0.0));
    }
}
