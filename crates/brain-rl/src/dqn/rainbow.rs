//! # Rainbow-Lite DQN
//!
//! Combines Double DQN, Dueling Network Architecture, and Prioritized Replay.
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

use super::super::buffer::PrioritizedReplayBuffer;
use super::super::core::{RlResult, Transition};
use super::super::policy::{EpsilonGreedyPolicy, EpsilonSchedule};
use super::dueling::DuelingQNet;
use super::DqnConfig;
use brain_core::Tensor;

/// Rainbow-Lite DQN Agent.
#[derive(Debug, Clone)]
pub struct RainbowAgent {
    pub q_online: DuelingQNet,
    pub q_target: DuelingQNet,
    pub per_buffer: PrioritizedReplayBuffer,
    pub policy: EpsilonGreedyPolicy,
    pub config: DqnConfig,
    pub total_steps: usize,
}

impl RainbowAgent {
    pub fn new(input_dim: usize, num_actions: usize, config: DqnConfig) -> Self {
        let q_online = DuelingQNet::new(input_dim, num_actions);
        let q_target = q_online.clone();
        let per_buffer = PrioritizedReplayBuffer::new(config.buffer_capacity, 0.6, 0.4);
        let schedule = EpsilonSchedule::Linear {
            start: config.epsilon_start,
            end: config.epsilon_end,
            decay_steps: config.epsilon_decay_steps,
        };
        let policy = EpsilonGreedyPolicy::new(schedule, num_actions);

        Self {
            q_online,
            q_target,
            per_buffer,
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
        self.per_buffer.push(transition);
        self.total_steps += 1;

        if self.per_buffer.buffer.len() < self.config.batch_size {
            return Ok(0.0);
        }

        let (tree_indices, batch, weights) =
            self.per_buffer.sample_batch(self.config.batch_size)?;
        let mut total_loss = 0.0;
        let gamma = self.config.gamma;

        for (i, t) in batch.iter().enumerate() {
            let q_current = self.q_online.forward(&t.state)[t.action];
            let online_next_q = self.q_online.forward(&t.next_state);
            let mut best_action = 0;
            let mut best_q = f64::NEG_INFINITY;
            for (a, &v) in online_next_q.iter().enumerate() {
                if v > best_q {
                    best_q = v;
                    best_action = a;
                }
            }

            let target_next_q = self.q_target.forward(&t.next_state)[best_action];
            let target = if t.done {
                t.reward
            } else {
                t.reward + gamma * target_next_q
            };
            let td_error = target - q_current;
            let loss = weights[i] * td_error * td_error;
            total_loss += loss;

            let new_priority = td_error.abs().max(1e-5).powf(self.per_buffer.alpha);
            self.per_buffer.tree.update(tree_indices[i], new_priority);
        }

        if self.total_steps % self.config.target_update_freq == 0 {
            self.q_target = self.q_online.clone();
        }

        Ok(total_loss / self.config.batch_size as f64)
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
