//! # Action-Value Q(s, a) Functions
//!
//! Tabular state-action tables and parameterized linear Q-networks.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use std::collections::HashMap;
use brain_core::Tensor;

/// Tabular State-Action Value Table Q(s, a).
#[derive(Debug, Clone, Default)]
pub struct QTable {
    pub table: HashMap<(usize, usize), f64>,
}

impl QTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, state: usize, action: usize) -> f64 {
        *self.table.get(&(state, action)).unwrap_or(&0.0)
    }

    pub fn set(&mut self, state: usize, action: usize, val: f64) {
        self.table.insert((state, action), val);
    }

    /// Computes max Q value across all actions.
    pub fn max_q(&self, state: usize, num_actions: usize) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for a in 0..num_actions {
            let q = self.get(state, a);
            if q > best { best = q; }
        }
        if best.is_infinite() { 0.0 } else { best }
    }

    /// Updates Q-value via standard Q-learning TD error.
    pub fn update_q(&mut self, s: usize, a: usize, r: f64, next_s: usize, num_actions: usize, gamma: f64, alpha: f64) {
        let target = r + gamma * self.max_q(next_s, num_actions);
        let current = self.get(s, a);
        self.set(s, a, current + alpha * (target - current));
    }
}

/// Multi-action Linear Q-Network.
#[derive(Debug, Clone)]
pub struct QNet {
    pub input_dim: usize,
    pub num_actions: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

impl QNet {
    pub fn new(input_dim: usize, num_actions: usize) -> Self {
        Self {
            input_dim,
            num_actions,
            weights: vec![0.0; input_dim * num_actions],
            biases: vec![0.0; num_actions],
        }
    }

    /// Evaluates Q-values for all discrete actions given state tensor.
    pub fn forward(&self, state: &Tensor) -> Vec<f64> {
        let d = state.data();
        let mut q = self.biases.clone();
        for a in 0..self.num_actions {
            for i in 0..d.len().min(self.input_dim) {
                q[a] += d[i] * self.weights[a * self.input_dim + i];
            }
        }
        q
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::env::*;
    use crate::policy::*;
    use crate::value::*;
    use crate::buffer::*;
    use crate::dqn::*;
    use crate::ppo::*;
    use crate::a2c::*;
    use crate::actor_critic::*;
    use crate::sac::*;
    use crate::agents::*;
    use crate::trainer::*;
    use crate::eval::*;
    use crate::checkpoint::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
