//! # Brain Reinforcement Learning Framework (`brain-rl`)
//!
//! Production-grade RL framework: DQN family, PPO, A2C, SAC, Actor-Critic, Environments, and Replay Buffers.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod core;
pub mod world_model;
pub mod curiosity;
pub mod skill_library;
pub mod env;
pub mod policy;
pub mod value;
pub mod buffer;
pub mod dqn;
pub mod ppo;
pub mod a2c;
pub mod actor_critic;
pub mod sac;
pub mod agents;
pub mod trainer;
pub mod eval;
pub mod checkpoint;
pub mod utils;

pub use core::{RlError, RlResult, Space, Trajectory, Transition};
pub use world_model::{WorldModel, WorldModelPrediction};
pub use curiosity::IntrinsicCuriosityModule;
pub use skill_library::{SkillLibrary, Skill, SkillStep};
pub use env::{CartPoleEnv, Env, EnvStep, GridWorldEnv, MountainCarEnv, PendulumEnv};
pub use policy::{CategoricalDist, DiagonalGaussianDist, EpsilonGreedyPolicy, EpsilonSchedule, GaussianPolicy, Policy};
pub use value::{QNet, QTable, VNet, VTable, ValueFn};
pub use buffer::{BufferStats, NStepBuffer, PrioritizedReplayBuffer, ReplayBuffer, SumTree, TrajectoryBuffer};
pub use dqn::{DqnAgent, DqnConfig, DoubleDqnAgent, DuelingDqnAgent, RainbowAgent};
pub use ppo::{PpoAgent, PpoClippedObjective, PpoConfig};
pub use a2c::{A2cAgent, A2cConfig};
pub use actor_critic::{ActorCriticNet, compute_gae};
pub use sac::{SacAgent, SacConfig};
pub use agents::{Agent, AgentKind, make_agent};
pub use trainer::{DqnTrainer, TrainerConfig};
pub use eval::{EvalReport, evaluate_dqn};
pub use checkpoint::RlCheckpoint;
pub use utils::{discount_returns, moving_average};

/// Semantic version of the `brain-rl` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and agents.
pub mod prelude {
    pub use super::core::{RlError, RlResult, Space, Trajectory, Transition};
    pub use super::env::{CartPoleEnv, Env, EnvStep, GridWorldEnv};
    pub use super::policy::{EpsilonGreedyPolicy, EpsilonSchedule, Policy};
    pub use super::value::{QTable, VTable, ValueFn};
    pub use super::buffer::{PrioritizedReplayBuffer, ReplayBuffer};
    pub use super::dqn::{DqnAgent, DqnConfig, DoubleDqnAgent, DuelingDqnAgent, RainbowAgent};
    pub use super::ppo::{PpoAgent, PpoConfig};
    pub use super::a2c::{A2cAgent, A2cConfig};
    pub use super::sac::{SacAgent, SacConfig};
    pub use super::agents::{Agent, AgentKind, make_agent};
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
