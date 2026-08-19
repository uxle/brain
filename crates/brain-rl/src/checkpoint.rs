//! # RL Agent Checkpointing & Serialization
//!
//! Checkpoints network parameters, replay buffer statistics, and exploration schedules.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::core::{RlError, RlResult};
use super::dqn::DqnAgent;

/// Serialized RL checkpoint representation.
#[derive(Debug, Clone, PartialEq)]
pub struct RlCheckpoint {
    pub total_steps: usize,
    pub q_weights: Vec<f64>,
    pub q_biases: Vec<f64>,
    pub buffer_count: usize,
}

impl RlCheckpoint {
    /// Creates checkpoint snapshot from active DQN agent.
    pub fn save_dqn(agent: &DqnAgent) -> Self {
        Self {
            total_steps: agent.total_steps,
            q_weights: agent.q_online.weights.clone(),
            q_biases: agent.q_online.biases.clone(),
            buffer_count: agent.buffer.len(),
        }
    }

    /// Restores saved parameters into a target DQN agent.
    pub fn load_dqn(&self, agent: &mut DqnAgent) -> RlResult<()> {
        if agent.q_online.weights.len() != self.q_weights.len() {
            return Err(RlError::CheckpointError("Weight shape mismatch".into()));
        }
        agent.total_steps = self.total_steps;
        agent.q_online.weights = self.q_weights.clone();
        agent.q_online.biases = self.q_biases.clone();
        agent.q_target = agent.q_online.clone();
        Ok(())
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
