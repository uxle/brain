//! # Synchronous Advantage Actor-Critic (A2C)
//!
//! Synchronous multi-environment trajectory collection and policy gradient advantage updates.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::policy::CategoricalDist;

/// Configuration hyperparameters for A2C.
#[derive(Debug, Clone, PartialEq)]
pub struct A2cConfig {
    pub lr: f64,
    pub gamma: f64,
    pub entropy_coef: f64,
    pub value_loss_coef: f64,
}

impl Default for A2cConfig {
    fn default() -> Self {
        Self {
            lr: 7e-4,
            gamma: 0.99,
            entropy_coef: 0.01,
            value_loss_coef: 0.5,
        }
    }
}

/// Synchronous Advantage Actor-Critic (A2C) Agent.
#[derive(Debug, Clone)]
pub struct A2cAgent {
    pub config: A2cConfig,
    pub input_dim: usize,
    pub num_actions: usize,
    pub actor_weights: Vec<f64>,
    pub critic_weights: Vec<f64>,
}

impl A2cAgent {
    pub fn new(input_dim: usize, num_actions: usize, config: A2cConfig) -> Self {
        Self {
            config,
            input_dim,
            num_actions,
            actor_weights: vec![0.0; input_dim * num_actions],
            critic_weights: vec![0.0; input_dim],
        }
    }

    pub fn act(&self, state: &Tensor) -> usize {
        let d = state.data();
        let mut logits = vec![0.0; self.num_actions];
        for a in 0..self.num_actions {
            for i in 0..d.len().min(self.input_dim) {
                logits[a] += d[i] * self.actor_weights[a * self.input_dim + i];
            }
        }
        let dist = CategoricalDist::from_logits(&logits);
        let mut best_a = 0;
        let mut best_p = f64::NEG_INFINITY;
        for (a, &p) in dist.probs.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_a = a;
            }
        }
        best_a
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
