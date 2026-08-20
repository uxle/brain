//! # Value Functions & Bellman Targets
//!
//! State Value V(s) tabular tables, neural network value representations, and target network updating.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

pub mod qvalue;

pub use qvalue::{QNet, QTable};

use brain_core::Tensor;
use std::collections::HashMap;

/// Fundamental trait for state value V(s) estimators.
pub trait ValueFn: Send + Sync {
    /// Predicts scalar state value V(s).
    fn value(&self, state: &Tensor) -> f64;

    /// Updates value function towards target value.
    fn update(&mut self, state: &Tensor, target: f64, lr: f64);
}

/// Tabular state value function V(s) using discrete state keys.
#[derive(Debug, Clone, Default)]
pub struct VTable {
    pub table: HashMap<usize, f64>,
}

impl VTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, state_idx: usize) -> f64 {
        *self.table.get(&state_idx).unwrap_or(&0.0)
    }

    pub fn set(&mut self, state_idx: usize, val: f64) {
        self.table.insert(state_idx, val);
    }

    pub fn update_td(&mut self, state_idx: usize, target: f64, lr: f64) {
        let current = self.get(state_idx);
        let updated = current + lr * (target - current);
        self.set(state_idx, updated);
    }
}

/// Linear or neural network state value estimator V(s).
#[derive(Debug, Clone)]
pub struct VNet {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl VNet {
    pub fn new(input_dim: usize) -> Self {
        Self {
            weights: vec![0.0; input_dim],
            bias: 0.0,
        }
    }

    pub fn forward(&self, state: &Tensor) -> f64 {
        let d = state.data();
        let mut sum = self.bias;
        for i in 0..d.len().min(self.weights.len()) {
            sum += d[i] * self.weights[i];
        }
        sum
    }

    pub fn train_step(&mut self, state: &Tensor, target: f64, lr: f64) {
        let pred = self.forward(state);
        let error = target - pred;
        let d = state.data();

        self.bias += lr * error;
        for i in 0..d.len().min(self.weights.len()) {
            self.weights[i] += lr * error * d[i];
        }
    }

    /// Polyak / soft target parameter update.
    pub fn soft_update_target(&mut self, online: &VNet, tau: f64) {
        self.bias = (1.0 - tau) * self.bias + tau * online.bias;
        for i in 0..self.weights.len() {
            self.weights[i] = (1.0 - tau) * self.weights[i] + tau * online.weights[i];
        }
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
    use crate::a2c::*;
    use crate::actor_critic::*;
    use crate::agents::*;
    use crate::buffer::*;
    use crate::checkpoint::*;
    use crate::core::*;
    use crate::dqn::*;
    use crate::env::*;
    use crate::eval::*;
    use crate::policy::*;
    use crate::ppo::*;
    use crate::sac::*;
    use crate::trainer::*;
    use crate::utils::*;
    use crate::value::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
