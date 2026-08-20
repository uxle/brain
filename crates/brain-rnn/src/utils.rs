//! # Numerical & Initialization Utilities
//!
//! Non-linear activations, orthogonal matrix generation, and gradient clipping.
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

use brain_core::Tensor;

/// Computes standard sigmoid activation: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Derivative of hyperbolic tangent: $\tanh'(x) = 1 - \tanh^2(x)$.
pub fn tanh_prime(tanh_val: f64) -> f64 {
    1.0 - tanh_val * tanh_val
}

/// Deterministic XorShift64 pseudo-random generator.
#[derive(Debug, Clone)]
pub struct RnnRng {
    pub state: u64,
}

impl RnnRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdeadbeef_cafebabe } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + self.next_f64() * (high - low)
    }

    pub fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Generates approximate orthogonal weight matrix of shape [rows, cols].
pub fn init_orthogonal(rows: usize, cols: usize, seed: u64) -> Tensor {
    let mut rng = RnnRng::new(seed);
    let mut data = vec![0.0; rows * cols];
    for val in data.iter_mut() {
        *val = rng.standard_normal();
    }

    // Gram-Schmidt orthogonalization across rows
    for i in 0..rows.min(cols) {
        // Normalize row i
        let mut norm_sq = 0.0;
        for j in 0..cols {
            let v = data[i * cols + j];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt().max(1e-12);
        for j in 0..cols {
            data[i * cols + j] /= norm;
        }

        // Subtract projection from subsequent rows
        for k in (i + 1)..rows.min(cols) {
            let mut dot = 0.0;
            for j in 0..cols {
                dot += data[i * cols + j] * data[k * cols + j];
            }
            for j in 0..cols {
                data[k * cols + j] -= dot * data[i * cols + j];
            }
        }
    }

    Tensor::from_slice(&data, vec![rows, cols])
}

/// Initializes uniform random weights in range $[-k, k]$ where $k = 1 / \sqrt{\text{fan\_in}}$.
pub fn init_uniform(rows: usize, cols: usize, fan_in: usize, seed: u64) -> Tensor {
    let mut rng = RnnRng::new(seed);
    let bound = 1.0 / (fan_in as f64).max(1.0).sqrt();
    let mut data = vec![0.0; rows * cols];
    for val in data.iter_mut() {
        *val = rng.uniform(-bound, bound);
    }
    Tensor::from_slice(&data, vec![rows, cols])
}

/// Clips gradient vector norm to max threshold.
pub fn clip_grad_norm(grads: &mut [f64], max_norm: f64) -> f64 {
    let total_norm: f64 = grads.iter().map(|&g| g * g).sum::<f64>().sqrt();
    if total_norm > max_norm && total_norm > 1e-12 {
        let scale = max_norm / total_norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    total_norm
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
