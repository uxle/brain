//! # RL Training Loops & Progress Logging
//!
//! Integrated episode trainers running agent-environment interaction steps with metrics logging.
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

use super::core::{RlResult, Transition};
use super::dqn::DqnAgent;
use super::env::Env;

/// Configuration for RL trainer episode loop.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainerConfig {
    pub max_episodes: usize,
    pub max_steps_per_episode: usize,
    pub eval_freq: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_episodes: 100,
            max_steps_per_episode: 500,
            eval_freq: 10,
        }
    }
}

/// DQN Trainer executing standard episode iterations.
pub struct DqnTrainer<E: Env> {
    pub agent: DqnAgent,
    pub env: E,
    pub config: TrainerConfig,
    pub episode_rewards: Vec<f64>,
}

impl<E: Env> DqnTrainer<E> {
    pub fn new(agent: DqnAgent, env: E, config: TrainerConfig) -> Self {
        Self {
            agent,
            env,
            config,
            episode_rewards: Vec::new(),
        }
    }

    /// Executes full training run and returns vector of episodic cumulative rewards.
    pub fn train(&mut self) -> RlResult<Vec<f64>> {
        for _ in 0..self.config.max_episodes {
            let mut state = self.env.reset()?;
            let mut ep_reward = 0.0;

            for _ in 0..self.config.max_steps_per_episode {
                let action = self.agent.act(&state);
                let step = self.env.step(action)?;
                ep_reward += step.reward;

                let transition = Transition::new(
                    state,
                    action,
                    step.reward,
                    step.observation.clone(),
                    step.done || step.truncated,
                );

                self.agent.step(transition)?;
                state = step.observation;

                if step.done || step.truncated {
                    break;
                }
            }

            self.episode_rewards.push(ep_reward);
        }

        Ok(self.episode_rewards.clone())
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
