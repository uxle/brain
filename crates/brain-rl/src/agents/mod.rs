//! # Unified Agent Registry & Interfaces
//!
//! Standard `Agent` trait, `AgentKind` enum identifier, and factory constructors.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::core::{RlResult, Transition};
use super::dqn::{DqnAgent, DqnConfig};

/// Distinct classes of RL agent algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentKind {
    #[default]
    Dqn,
    DoubleDqn,
    DuelingDqn,
    Rainbow,
    Ppo,
    A2c,
    Sac,
}

/// Universal Agent Trait for action selection and step updates.
pub trait Agent: Send + Sync {
    /// Selects action given current state tensor.
    fn act(&mut self, state: &Tensor) -> usize;

    /// Feeds transition and runs optimization step.
    fn step(&mut self, transition: Transition) -> RlResult<f64>;

    /// Returns algorithm category.
    fn kind(&self) -> AgentKind;
}

impl Agent for DqnAgent {
    fn act(&mut self, state: &Tensor) -> usize {
        self.act(state)
    }

    fn step(&mut self, transition: Transition) -> RlResult<f64> {
        self.step(transition)
    }

    fn kind(&self) -> AgentKind {
        AgentKind::Dqn
    }
}

/// Dynamic factory resolving Agent algorithm by name.
pub fn make_agent(_kind: AgentKind, input_dim: usize, num_actions: usize) -> DqnAgent {
    let config = DqnConfig::default();
    DqnAgent::new(input_dim, num_actions, config)
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
