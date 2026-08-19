//! # Double Deep Q-Networks (Double DQN)
//!
//! Decouples action selection (online network) from action evaluation (target network) to prevent overestimation bias.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::buffer::ReplayBuffer;
use super::super::core::{RlResult, Transition};
use super::super::policy::{EpsilonGreedyPolicy, EpsilonSchedule};
use super::super::value::QNet;
use super::DqnConfig;

/// Double DQN Agent.
#[derive(Debug, Clone)]
pub struct DoubleDqnAgent {
    pub q_online: QNet,
    pub q_target: QNet,
    pub buffer: ReplayBuffer,
    pub policy: EpsilonGreedyPolicy,
    pub config: DqnConfig,
    pub total_steps: usize,
}

impl DoubleDqnAgent {
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

    pub fn act(&mut self, state: &Tensor) -> usize {
        let q_values = self.q_online.forward(state);
        self.policy.select_action(&q_values, self.total_steps)
    }

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
            let online_next_q = self.q_online.forward(&t.next_state);
            let mut best_action = 0;
            let mut best_online_q = f64::NEG_INFINITY;
            for (a, &v) in online_next_q.iter().enumerate() {
                if v > best_online_q {
                    best_online_q = v;
                    best_action = a;
                }
            }

            let target_next_q = self.q_target.forward(&t.next_state)[best_action];
            let target = if t.done { t.reward } else { t.reward + gamma * target_next_q };
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
