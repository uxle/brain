//! # Soft Actor-Critic (SAC)
//!
//! Off-policy actor-critic with maximum entropy objective and twin Q-functions.
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

/// Configuration hyperparameters for Soft Actor-Critic.
#[derive(Debug, Clone, PartialEq)]
pub struct SacConfig {
    pub gamma: f64,
    pub tau: f64,
    pub alpha: f64,
    pub auto_entropy_tuning: bool,
    pub target_entropy: f64,
    pub lr: f64,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            tau: 0.005,
            alpha: 0.2,
            auto_entropy_tuning: true,
            target_entropy: -1.0,
            lr: 3e-4,
        }
    }
}

/// Soft Actor-Critic (SAC) Agent.
#[derive(Debug, Clone)]
pub struct SacAgent {
    pub config: SacConfig,
    pub state_dim: usize,
    pub action_dim: usize,
    pub log_alpha: f64,
}

impl SacAgent {
    pub fn new(state_dim: usize, action_dim: usize, config: SacConfig) -> Self {
        let log_alpha = config.alpha.ln();
        Self {
            config,
            state_dim,
            action_dim,
            log_alpha,
        }
    }

    pub fn alpha(&self) -> f64 {
        self.log_alpha.exp()
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
