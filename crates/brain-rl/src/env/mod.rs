//! # Environment Abstractions & Step Definitions
//!
//! Standard `Env` trait, step responses, and information dictionaries.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod gym;
pub mod gridworld;
pub mod atari_lite;
pub mod mujoco_lite;
pub mod wrappers;
pub mod vector;

pub use gym::{CartPoleEnv, MountainCarEnv, PendulumEnv};
pub use gridworld::{GridWorldEnv, CliffWalkingEnv, FrozenLakeEnv};
pub use atari_lite::{PongLiteEnv, BreakoutLiteEnv};
pub use mujoco_lite::{HalfCheetahLiteEnv, ReacherLiteEnv};
pub use wrappers::{FrameStackWrapper, TimeLimitWrapper, RewardScaleWrapper};
pub use vector::{VecEnv, DummyVecEnv};

use brain_core::Tensor;
use super::core::{RlResult, Space};

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
