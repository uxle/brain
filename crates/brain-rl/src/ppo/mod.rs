//! # Proximal Policy Optimization (PPO)
//!
//! Clipped surrogate objective policy optimization with trajectory rollout memory.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod clipped;

pub use clipped::PpoClippedObjective;

use brain_core::Tensor;
use super::core::Trajectory;
use super::policy::CategoricalDist;

/// Configuration hyperparameters for PPO.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoConfig {
    pub clip_ratio: f64,
    pub lr_actor: f64,
    pub lr_critic: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub ppo_epochs: usize,
    pub batch_size: usize,
    pub entropy_coef: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_ratio: 0.2,
            lr_actor: 3e-4,
            lr_critic: 1e-3,
            gamma: 0.99,
            gae_lambda: 0.95,
            ppo_epochs: 10,
            batch_size: 64,
            entropy_coef: 0.01,
        }
    }
}

/// PPO Agent.
#[derive(Debug, Clone)]
pub struct PpoAgent {
    pub config: PpoConfig,
    pub input_dim: usize,
    pub num_actions: usize,
    pub actor_weights: Vec<f64>,
    pub critic_weights: Vec<f64>,
    pub trajectory: Trajectory,
}

impl PpoAgent {
    pub fn new(input_dim: usize, num_actions: usize, config: PpoConfig) -> Self {
        Self {
            config,
            input_dim,
            num_actions,
            actor_weights: vec![0.0; input_dim * num_actions],
            critic_weights: vec![0.0; input_dim],
            trajectory: Trajectory::new(),
        }
    }

    pub fn get_logits(&self, state: &Tensor) -> Vec<f64> {
        let d = state.data();
        let mut logits = vec![0.0; self.num_actions];
        for a in 0..self.num_actions {
            for i in 0..d.len().min(self.input_dim) {
                logits[a] += d[i] * self.actor_weights[a * self.input_dim + i];
            }
        }
        logits
    }

    pub fn get_value(&self, state: &Tensor) -> f64 {
        let d = state.data();
        let mut v = 0.0;
        for i in 0..d.len().min(self.input_dim) {
            v += d[i] * self.critic_weights[i];
        }
        v
    }

    pub fn act(&self, state: &Tensor) -> (usize, f64) {
        let logits = self.get_logits(state);
        let dist = CategoricalDist::from_logits(&logits);
        let mut best_a = 0;
        let mut best_p = f64::NEG_INFINITY;
        for (a, &p) in dist.probs.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_a = a;
            }
        }
        (best_a, dist.log_prob(best_a))
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
