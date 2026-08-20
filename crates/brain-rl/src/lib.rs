//! # Brain Reinforcement Learning Framework (`brain-rl`)
//!
//! Production-grade RL framework: DQN family, PPO, A2C, SAC, Actor-Critic, Environments, and Replay Buffers.
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

pub mod a2c;
pub mod actor_critic;
pub mod agents;
pub mod buffer;
pub mod checkpoint;
pub mod core;
pub mod curiosity;
pub mod dqn;
pub mod env;
pub mod eval;
pub mod policy;
pub mod ppo;
pub mod sac;
pub mod skill_library;
pub mod trainer;
pub mod utils;
pub mod value;
pub mod world_model;

pub use a2c::{A2cAgent, A2cConfig};
pub use actor_critic::{compute_gae, ActorCriticNet};
pub use agents::{make_agent, Agent, AgentKind};
pub use buffer::{
    BufferStats, NStepBuffer, PrioritizedReplayBuffer, ReplayBuffer, SumTree, TrajectoryBuffer,
};
pub use checkpoint::RlCheckpoint;
pub use core::{RlError, RlResult, Space, Trajectory, Transition};
pub use curiosity::IntrinsicCuriosityModule;
pub use dqn::{DoubleDqnAgent, DqnAgent, DqnConfig, DuelingDqnAgent, RainbowAgent};
pub use env::{CartPoleEnv, Env, EnvStep, GridWorldEnv, MountainCarEnv, PendulumEnv};
pub use eval::{evaluate_dqn, EvalReport};
pub use policy::{
    CategoricalDist, DiagonalGaussianDist, EpsilonGreedyPolicy, EpsilonSchedule, GaussianPolicy,
    Policy,
};
pub use ppo::{PpoAgent, PpoClippedObjective, PpoConfig};
pub use sac::{SacAgent, SacConfig};
pub use skill_library::{Skill, SkillLibrary, SkillStep};
pub use trainer::{DqnTrainer, TrainerConfig};
pub use utils::{discount_returns, moving_average};
pub use value::{QNet, QTable, VNet, VTable, ValueFn};
pub use world_model::{WorldModel, WorldModelPrediction};

/// Semantic version of the `brain-rl` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and agents.
pub mod prelude {
    pub use super::a2c::{A2cAgent, A2cConfig};
    pub use super::agents::{make_agent, Agent, AgentKind};
    pub use super::buffer::{PrioritizedReplayBuffer, ReplayBuffer};
    pub use super::core::{RlError, RlResult, Space, Trajectory, Transition};
    pub use super::dqn::{DoubleDqnAgent, DqnAgent, DqnConfig, DuelingDqnAgent, RainbowAgent};
    pub use super::env::{CartPoleEnv, Env, EnvStep, GridWorldEnv};
    pub use super::policy::{EpsilonGreedyPolicy, EpsilonSchedule, Policy};
    pub use super::ppo::{PpoAgent, PpoConfig};
    pub use super::sac::{SacAgent, SacConfig};
    pub use super::value::{QTable, VTable, ValueFn};
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
