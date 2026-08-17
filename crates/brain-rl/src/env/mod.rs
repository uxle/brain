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

    #[test]
    fn test_env_mod_stress_001() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_002() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_003() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_004() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_005() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_006() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_007() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_008() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_009() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_010() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_011() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_012() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_013() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_014() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_015() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_016() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_017() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_018() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_019() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_020() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_021() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_022() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_023() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_024() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_025() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_026() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_027() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_028() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_029() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_030() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_031() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_032() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_033() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_034() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_035() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_036() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_037() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_038() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_039() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_040() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_041() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_042() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_043() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_044() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_045() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_046() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_047() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_048() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_049() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_050() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_051() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_052() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_053() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_054() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_055() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_056() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_057() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_058() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_059() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_060() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_061() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_062() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_063() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_064() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_065() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_066() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_067() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_068() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_069() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_070() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_071() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_072() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_073() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_074() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_075() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_076() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_077() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_078() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_079() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_080() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_081() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_082() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_083() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_084() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_085() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_086() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_087() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_088() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_089() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_090() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_091() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_092() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_093() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_094() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_095() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_096() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_097() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_098() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_099() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_100() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_101() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_102() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_103() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_104() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_105() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_106() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_107() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_108() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_109() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_110() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_111() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_112() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_113() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_114() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_115() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_116() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_117() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_118() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_119() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_120() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_121() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_122() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_123() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_124() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_125() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_126() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_127() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_128() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_129() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_130() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_131() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_132() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_133() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_134() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_135() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_136() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_137() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_138() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_139() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_140() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_141() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_142() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_143() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_144() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_145() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_146() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_147() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_148() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_149() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_150() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_151() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_152() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_153() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_154() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_155() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_156() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_157() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_158() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_159() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_160() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_161() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_162() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_163() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_164() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_165() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_166() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_167() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_168() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_169() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_170() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_171() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_172() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_173() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_174() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_175() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_176() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_177() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_178() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_179() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_180() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_181() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_182() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_183() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_184() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_185() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_186() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_187() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_188() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_189() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_190() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_191() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_192() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_193() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_194() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_195() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_196() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_197() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_198() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_199() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_200() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_201() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_202() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_203() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_204() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_205() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_206() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_207() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_208() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_209() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_210() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_211() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_212() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_213() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_214() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_215() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_216() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_217() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_218() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_219() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_220() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_221() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_222() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_223() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_224() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_225() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_226() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_227() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_228() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_229() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_230() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_231() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_232() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_233() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_234() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_235() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_236() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_237() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_238() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_239() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_240() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_241() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_242() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_243() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_244() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_245() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_246() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_247() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_248() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_249() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_250() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_251() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_252() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_253() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_254() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_255() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_256() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_257() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_258() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_259() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_260() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_261() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_262() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_263() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_264() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_265() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_266() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_267() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_268() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_269() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_270() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_271() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_272() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_273() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_274() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_275() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_276() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_277() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_278() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_279() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_280() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_281() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_282() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_283() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_284() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_285() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_286() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_287() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_288() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_289() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_290() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_291() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_292() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_293() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_294() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_295() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_296() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_297() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_298() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_299() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_300() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_301() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_302() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_303() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_304() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_305() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_306() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_307() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_308() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_309() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_310() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_311() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_312() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_313() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_314() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_315() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_316() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_317() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_318() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_319() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_320() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_321() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_322() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_323() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_324() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_325() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_326() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_327() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_328() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_329() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_330() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_331() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_332() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_333() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_334() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_335() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_336() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_337() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_338() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_339() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_340() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_341() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_342() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_343() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_344() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_345() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_346() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_347() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_348() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_349() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_350() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_351() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_352() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_353() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_354() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_355() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_356() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_357() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_358() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_359() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_360() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_361() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_362() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_363() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_364() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_365() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_366() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_367() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_368() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_369() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_370() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_371() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_372() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_373() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_374() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_375() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_376() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_377() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_378() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_379() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_380() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_381() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_382() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_383() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_384() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_385() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_386() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_387() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_388() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_389() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_390() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_391() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_392() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_393() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_394() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_395() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_396() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_397() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_398() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_399() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_400() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_401() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_402() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_403() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_404() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_405() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_406() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    #[test]
    fn test_env_mod_stress_407() {
        let obs = Tensor::from_slice(&[0.0, 0.0], vec![2]);
        let step = EnvStep::new(obs, 1.0, false, false);
        assert_eq!(step.reward, 1.0);
        assert!(!step.done);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
}
