//! # Environment Abstractions & Step Definitions
//!
//! Standard `Env` trait, step responses, and information dictionaries.
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

pub mod atari_lite;
pub mod gridworld;
pub mod gym;
pub mod mujoco_lite;
pub mod vector;
pub mod wrappers;

pub use atari_lite::{BreakoutLiteEnv, PongLiteEnv};
pub use gridworld::{CliffWalkingEnv, FrozenLakeEnv, GridWorldEnv};
pub use gym::{CartPoleEnv, MountainCarEnv, PendulumEnv};
pub use mujoco_lite::{HalfCheetahLiteEnv, ReacherLiteEnv};
pub use vector::{DummyVecEnv, VecEnv};
pub use wrappers::{FrameStackWrapper, RewardScaleWrapper, TimeLimitWrapper};

use super::core::{RlResult, Space};
use brain_core::Tensor;

/// Step output container returned from environment transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvStep {
    pub observation: Tensor,
    pub reward: f64,
    pub done: bool,
    pub truncated: bool,
}

impl EnvStep {
    pub fn new(observation: Tensor, reward: f64, done: bool, truncated: bool) -> Self {
        Self {
            observation,
            reward,
            done,
            truncated,
        }
    }
}

/// Environment trait for standard RL simulation environments.
pub trait Env: Send + Sync {
    /// Resets environment to initial state and returns initial observation.
    fn reset(&mut self) -> RlResult<Tensor>;

    /// Executes discrete action step and returns transition result.
    fn step(&mut self, action: usize) -> RlResult<EnvStep>;

    /// Returns observation state space specification.
    fn observation_space(&self) -> Space;

    /// Returns action space specification.
    fn action_space(&self) -> Space;

    /// Renders environment state as ASCII text representation.
    fn render_ascii(&self) -> String {
        "Env State".to_string()
    }

    /// Seeds internal pseudo-random number generator.
    fn seed(&mut self, _seed: u64) {}
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
