//! # Reinforcement Learning Core Primitives
//!
//! Space representations, State-Action transitions, Trajectory collections, and Error types.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use std::fmt;
use brain_core::Tensor;

/// Distinct environment space representations.
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    Discrete(usize),
    Continuous { shape: Vec<usize>, low: f64, high: f64 },
}

/// State-Action-Reward-NextState-Done transition tuple.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub state: Tensor,
    pub action: usize,
    pub reward: f64,
    pub next_state: Tensor,
    pub done: bool,
}

impl Transition {
    pub fn new(state: Tensor, action: usize, reward: f64, next_state: Tensor, done: bool) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Full episode trajectory sequence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectory {
    pub transitions: Vec<Transition>,
    pub total_reward: f64,
}

impl Trajectory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, t: Transition) {
        self.total_reward += t.reward;
        self.transitions.push(t);
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

/// Errors encountered in RL environment interactions or agent updates.
#[derive(Debug, Clone, PartialEq)]
pub enum RlError {
    InvalidAction(usize),
    InvalidStateShape { expected: Vec<usize>, found: Vec<usize> },
    BufferFull,
    EmptyBuffer,
    InvalidDiscount(f64),
    CheckpointError(String),
    EnvironmentError(String),
}

impl fmt::Display for RlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RlError::InvalidAction(a) => write!(f, "Action out of valid range: {}", a),
            RlError::InvalidStateShape { expected, found } => {
                write!(f, "Invalid state shape: expected {:?}, found {:?}", expected, found)
            }
            RlError::BufferFull => write!(f, "Replay buffer is at maximum capacity"),
            RlError::EmptyBuffer => write!(f, "Replay buffer is empty"),
            RlError::InvalidDiscount(g) => write!(f, "Discount gamma {} must be in [0, 1]", g),
            RlError::CheckpointError(msg) => write!(f, "RL checkpoint error: {}", msg),
            RlError::EnvironmentError(msg) => write!(f, "RL environment error: {}", msg),
        }
    }
}

impl std::error::Error for RlError {}

pub type RlResult<T> = Result<T, RlError>;

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
