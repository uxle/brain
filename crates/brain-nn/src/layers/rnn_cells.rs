//! # Recurrent Gate Cells (LSTMCell & GRUCell)
//!
//! Single-step recurrent gate computation cells for custom unrolled sequence loops.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for recurrent cell initialization.
#[derive(Debug, Clone, Default)]
pub struct CellConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

/// Single-step LSTM cell.
#[derive(Debug, Clone)]
pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias: Option<Tensor>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let bound = (1.0 / (hidden_size as f64).max(1.0)).sqrt();
        let num_ih = 4 * hidden_size * input_size;
        let mut w_ih = Vec::with_capacity(num_ih);
        for i in 0..num_ih {
            w_ih.push(((i as f64 * 0.173).sin()) * bound);
        }
        let num_hh = 4 * hidden_size * hidden_size;
        let mut w_hh = Vec::with_capacity(num_hh);
        for i in 0..num_hh {
            w_hh.push(((i as f64 * 0.281).sin()) * bound);
        }

        Self {
            input_size,
            hidden_size,
            weight_ih: Tensor::from_vec(w_ih, vec![4 * hidden_size, input_size]),
            weight_hh: Tensor::from_vec(w_hh, vec![4 * hidden_size, hidden_size]),
            bias: Some(Tensor::zeros(vec![4 * hidden_size])),
        }
    }

    pub fn forward_step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        let b = x.shape()[0];
        let h_dim = self.hidden_size;
        let x_data = x.to_vec();
        let h_data = h.to_vec();
        let c_data = c.to_vec();
        let w_ih = self.weight_ih.to_vec();
        let w_hh = self.weight_hh.to_vec();
        let bias = self.bias.as_ref().map(|b| b.to_vec());

        let mut next_h = vec![0.0; b * h_dim];
        let mut next_c = vec![0.0; b * h_dim];

        for bi in 0..b {
            let x_slice = &x_data[bi * self.input_size..(bi + 1) * self.input_size];
            let h_slice = &h_data[bi * h_dim..(bi + 1) * h_dim];
            let c_slice = &c_data[bi * h_dim..(bi + 1) * h_dim];

            for j in 0..h_dim {
                let mut gate_i = bias.as_ref().map(|b| b[j]).unwrap_or(0.0);
                let mut gate_f = bias.as_ref().map(|b| b[h_dim + j]).unwrap_or(0.0);
                let mut gate_g = bias.as_ref().map(|b| b[2 * h_dim + j]).unwrap_or(0.0);
                let mut gate_o = bias.as_ref().map(|b| b[3 * h_dim + j]).unwrap_or(0.0);

                for k in 0..self.input_size {
                    gate_i += x_slice[k] * w_ih[j * self.input_size + k];
                    gate_f += x_slice[k] * w_ih[(h_dim + j) * self.input_size + k];
                    gate_g += x_slice[k] * w_ih[(2 * h_dim + j) * self.input_size + k];
                    gate_o += x_slice[k] * w_ih[(3 * h_dim + j) * self.input_size + k];
                }
                for k in 0..h_dim {
                    gate_i += h_slice[k] * w_hh[j * h_dim + k];
                    gate_f += h_slice[k] * w_hh[(h_dim + j) * h_dim + k];
                    gate_g += h_slice[k] * w_hh[(2 * h_dim + j) * h_dim + k];
                    gate_o += h_slice[k] * w_hh[(3 * h_dim + j) * h_dim + k];
                }

                let i_act = sigmoid(gate_i);
                let f_act = sigmoid(gate_f);
                let g_act = gate_g.tanh();
                let o_act = sigmoid(gate_o);

                let c_val = f_act * c_slice[j] + i_act * g_act;
                let h_val = o_act * c_val.tanh();

                next_c[bi * h_dim + j] = c_val;
                next_h[bi * h_dim + j] = h_val;
            }
        }

        (
            Tensor::from_vec(next_h, vec![b, h_dim]),
            Tensor::from_vec(next_c, vec![b, h_dim]),
        )
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}

/// Single-step GRU cell.
#[derive(Debug, Clone)]
pub struct GRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias: Option<Tensor>,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let bound = (1.0 / (hidden_size as f64).max(1.0)).sqrt();
        let num_ih = 3 * hidden_size * input_size;
        let mut w_ih = Vec::with_capacity(num_ih);
        for i in 0..num_ih {
            w_ih.push(((i as f64 * 0.173).sin()) * bound);
        }
        let num_hh = 3 * hidden_size * hidden_size;
        let mut w_hh = Vec::with_capacity(num_hh);
        for i in 0..num_hh {
            w_hh.push(((i as f64 * 0.281).sin()) * bound);
        }

        Self {
            input_size,
            hidden_size,
            weight_ih: Tensor::from_vec(w_ih, vec![3 * hidden_size, input_size]),
            weight_hh: Tensor::from_vec(w_hh, vec![3 * hidden_size, hidden_size]),
            bias: Some(Tensor::zeros(vec![3 * hidden_size])),
        }
    }

    pub fn forward_step(&self, x: &Tensor, h: &Tensor) -> Tensor {
        let b = x.shape()[0];
        let h_dim = self.hidden_size;
        let x_data = x.to_vec();
        let h_data = h.to_vec();
        let w_ih = self.weight_ih.to_vec();
        let w_hh = self.weight_hh.to_vec();
        let bias = self.bias.as_ref().map(|b| b.to_vec());

        let mut next_h = vec![0.0; b * h_dim];

        for bi in 0..b {
            let x_slice = &x_data[bi * self.input_size..(bi + 1) * self.input_size];
            let h_slice = &h_data[bi * h_dim..(bi + 1) * h_dim];

            for j in 0..h_dim {
                let mut gate_r = bias.as_ref().map(|b| b[j]).unwrap_or(0.0);
                let mut gate_z = bias.as_ref().map(|b| b[h_dim + j]).unwrap_or(0.0);
                let mut gate_n_x = bias.as_ref().map(|b| b[2 * h_dim + j]).unwrap_or(0.0);
                let mut gate_n_h = 0.0;

                for k in 0..self.input_size {
                    gate_r += x_slice[k] * w_ih[j * self.input_size + k];
                    gate_z += x_slice[k] * w_ih[(h_dim + j) * self.input_size + k];
                    gate_n_x += x_slice[k] * w_ih[(2 * h_dim + j) * self.input_size + k];
                }
                for k in 0..h_dim {
                    gate_r += h_slice[k] * w_hh[j * h_dim + k];
                    gate_z += h_slice[k] * w_hh[(h_dim + j) * h_dim + k];
                    gate_n_h += h_slice[k] * w_hh[(2 * h_dim + j) * h_dim + k];
                }

                let r_act = sigmoid(gate_r);
                let z_act = sigmoid(gate_z);
                let n_act = (gate_n_x + r_act * gate_n_h).tanh();

                let h_val = (1.0 - z_act) * n_act + z_act * h_slice[j];
                next_h[bi * h_dim + j] = h_val;
            }
        }

        Tensor::from_vec(next_h, vec![b, h_dim])
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_lstm_cell_step() {
        let cell = LSTMCell::new(4, 8);
        let x = Tensor::zeros(vec![2, 4]);
        let h = Tensor::zeros(vec![2, 8]);
        let c = Tensor::zeros(vec![2, 8]);
        let (next_h, next_c) = cell.forward_step(&x, &h, &c);
        assert_eq!(next_h.shape(), &[2, 8]);
        assert_eq!(next_c.shape(), &[2, 8]);
    }

    #[test]
    fn test_gru_cell_step() {
        let cell = GRUCell::new(4, 8);
        let x = Tensor::zeros(vec![2, 4]);
        let h = Tensor::zeros(vec![2, 8]);
        let next_h = cell.forward_step(&x, &h);
        assert_eq!(next_h.shape(), &[2, 8]);
    }
}
