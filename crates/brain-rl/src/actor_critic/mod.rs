//! # Shared Actor-Critic Backbones & Generalized Advantage Estimation (GAE)
//!
//! Generalized Advantage Estimation (GAE) recursive return discounting.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// Computes Generalized Advantage Estimation (GAE) returns and advantages over a trajectory.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    next_value: f64,
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut returns = vec![0.0; n];
    let mut last_gae = 0.0;

    for t in (0..n).rev() {
        let next_val = if t + 1 < n { values[t + 1] } else { next_value };
        let non_terminal = if dones[t] { 0.0 } else { 1.0 };
        let delta = rewards[t] + gamma * next_val * non_terminal - values[t];
        last_gae = delta + gamma * lambda * non_terminal * last_gae;
        advantages[t] = last_gae;
        returns[t] = advantages[t] + values[t];
    }

    (advantages, returns)
}

/// Unified Actor-Critic Neural Network Representation.
#[derive(Debug, Clone)]
pub struct ActorCriticNet {
    pub input_dim: usize,
    pub num_actions: usize,
    pub actor_weights: Vec<f64>,
    pub critic_weights: Vec<f64>,
}

impl ActorCriticNet {
    pub fn new(input_dim: usize, num_actions: usize) -> Self {
        Self {
            input_dim,
            num_actions,
            actor_weights: vec![0.0; input_dim * num_actions],
            critic_weights: vec![0.0; input_dim],
        }
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
