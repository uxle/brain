//! # Deep Q-Networks (DQN)
//!
//! Standard DQN agent with online/target Q-networks, replay buffer, and epsilon-greedy exploration.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod double;
pub mod dueling;
pub mod rainbow;

pub use double::DoubleDqnAgent;
pub use dueling::DuelingDqnAgent;
pub use rainbow::RainbowAgent;

use brain_core::Tensor;
use super::buffer::ReplayBuffer;
use super::core::{RlResult, Transition};
use super::policy::{EpsilonGreedyPolicy, EpsilonSchedule};
use super::value::QNet;

/// Configuration hyperparameters for DQN Agent.
#[derive(Debug, Clone, PartialEq)]
pub struct DqnConfig {
    pub gamma: f64,
    pub lr: f64,
    pub batch_size: usize,
    pub target_update_freq: usize,
    pub buffer_capacity: usize,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay_steps: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lr: 1e-3,
            batch_size: 32,
            target_update_freq: 100,
            buffer_capacity: 10000,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 1000,
        }
    }
}

/// Deep Q-Network Agent.
#[derive(Debug, Clone)]
pub struct DqnAgent {
    pub q_online: QNet,
    pub q_target: QNet,
    pub buffer: ReplayBuffer,
    pub policy: EpsilonGreedyPolicy,
    pub config: DqnConfig,
    pub total_steps: usize,
}

impl DqnAgent {
    pub fn new(input_dim: usize, num_actions: usize, config: DqnConfig) -> Self {
        let q_online = QNet::new(input_dim, num_actions);
        let q_target = q_online.clone();
        let buffer = ReplayBuffer::new(config.buffer_capacity);
        let schedule = EpsilonSchedule::Linear {
            start: config.epsilon_start,
            end: config.epsilon_end,
            decay_steps: config.epsilon_decay_steps,
        };
        let policy = EpsilonGreedyPolicy::new(schedule, num_actions);

        Self {
            q_online,
            q_target,
            buffer,
            policy,
            config,
            total_steps: 0,
        }
    }

    /// Selects action for given state observation.
    pub fn act(&mut self, state: &Tensor) -> usize {
        let q_values = self.q_online.forward(state);
        self.policy.select_action(&q_values, self.total_steps)
    }

    /// Stores transition into replay buffer and performs training update step.
    pub fn step(&mut self, transition: Transition) -> RlResult<f64> {
        self.buffer.push(transition);
        self.total_steps += 1;

        if self.buffer.len() < self.config.batch_size {
            return Ok(0.0);
        }

        let batch = self.buffer.sample_batch(self.config.batch_size)?;
        let mut total_loss = 0.0;
        let gamma = self.config.gamma;
        let lr = self.config.lr;

        for t in &batch {
            let q_current = self.q_online.forward(&t.state)[t.action];
            let q_next = self.q_target.forward(&t.next_state);
            let mut max_q_next = f64::NEG_INFINITY;
            for &v in &q_next {
                if v > max_q_next { max_q_next = v; }
            }
            if max_q_next.is_infinite() { max_q_next = 0.0; }

            let target = if t.done { t.reward } else { t.reward + gamma * max_q_next };
            let error = target - q_current;
            total_loss += error * error;

            let s_data = t.state.data();
            for i in 0..s_data.len().min(self.q_online.input_dim) {
                self.q_online.weights[t.action * self.q_online.input_dim + i] += lr * error * s_data[i];
            }
            self.q_online.biases[t.action] += lr * error;
        }

        if self.total_steps % self.config.target_update_freq == 0 {
            self.q_target = self.q_online.clone();
        }

        Ok(total_loss / self.config.batch_size as f64)
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
