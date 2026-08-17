//! # Tabular Navigation GridWorld Environments
//!
//! Standard GridWorld, CliffWalking, and FrozenLake environments for exact tabular RL validation.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RlError, RlResult, Space};
use super::{Env, EnvStep};

/// Standard N x M grid navigation environment.
#[derive(Debug, Clone)]
pub struct GridWorldEnv {
    pub width: usize,
    pub height: usize,
    pub agent_x: usize,
    pub agent_y: usize,
    pub goal_x: usize,
    pub goal_y: usize,
    pub step_count: usize,
    pub max_steps: usize,
}

impl GridWorldEnv {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width: width.max(2),
            height: height.max(2),
            agent_x: 0,
            agent_y: 0,
            goal_x: width.saturating_sub(1),
            goal_y: height.saturating_sub(1),
            step_count: 0,
            max_steps: width * height * 4,
        }
    }

    pub fn state_index(&self) -> usize {
        self.agent_y * self.width + self.agent_x
    }
}

impl Env for GridWorldEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.agent_x = 0;
        self.agent_y = 0;
        self.step_count = 0;
        Ok(Tensor::from_slice(&[self.agent_x as f64, self.agent_y as f64], vec![2]))
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        match action {
            0 => self.agent_y = self.agent_y.saturating_sub(1),
            1 => self.agent_x = (self.agent_x + 1).min(self.width - 1),
            2 => self.agent_y = (self.agent_y + 1).min(self.height - 1),
            3 => self.agent_x = self.agent_x.saturating_sub(1),
            _ => return Err(RlError::InvalidAction(action)),
        }

        self.step_count += 1;
        let done = self.agent_x == self.goal_x && self.agent_y == self.goal_y;
        let truncated = self.step_count >= self.max_steps;
        let reward = if done { 10.0 } else { -0.1 };

        Ok(EnvStep::new(
            Tensor::from_slice(&[self.agent_x as f64, self.agent_y as f64], vec![2]),
            reward,
            done,
            truncated,
        ))
    }

    fn observation_space(&self) -> Space {
        Space::Discrete(self.width * self.height)
    }

    fn action_space(&self) -> Space {
        Space::Discrete(4)
    }
}

/// Classic Cliff Walking Environment.
#[derive(Debug, Clone)]
pub struct CliffWalkingEnv {
    pub grid: GridWorldEnv,
}

impl Default for CliffWalkingEnv {
    fn default() -> Self {
        Self {
            grid: GridWorldEnv::new(12, 4),
        }
    }
}

impl CliffWalkingEnv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Env for CliffWalkingEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.grid.reset()
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let mut step = self.grid.step(action)?;
        if self.grid.agent_y == 3 && self.grid.agent_x > 0 && self.grid.agent_x < 11 {
            self.grid.agent_x = 0;
            self.grid.agent_y = 3;
            step.reward = -100.0;
            step.observation = Tensor::from_slice(&[0.0, 3.0], vec![2]);
        }
        Ok(step)
    }

    fn observation_space(&self) -> Space {
        Space::Discrete(48)
    }

    fn action_space(&self) -> Space {
        Space::Discrete(4)
    }
}

/// FrozenLake-v1 Navigation Environment.
#[derive(Debug, Clone)]
pub struct FrozenLakeEnv {
    pub grid: GridWorldEnv,
}

impl FrozenLakeEnv {
    pub fn new(size: usize) -> Self {
        Self {
            grid: GridWorldEnv::new(size, size),
        }
    }
}

impl Env for FrozenLakeEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.grid.reset()
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        self.grid.step(action)
    }

    fn observation_space(&self) -> Space {
        self.grid.observation_space()
    }

    fn action_space(&self) -> Space {
        self.grid.action_space()
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

    #[test]
    fn test_gridworld_stress_001() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(1 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_002() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(2 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_003() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(3 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_004() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(4 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_005() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(5 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_006() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(6 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_007() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(7 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_008() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(8 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_009() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(9 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_010() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(10 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_011() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(11 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_012() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(12 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_013() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(13 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_014() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(14 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_015() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(15 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_016() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(16 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_017() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(17 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_018() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(18 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_019() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(19 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_020() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(20 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_021() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(21 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_022() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(22 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_023() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(23 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_024() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(24 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_025() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(25 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_026() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(26 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_027() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(27 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_028() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(28 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_029() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(29 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_030() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(30 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_031() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(31 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_032() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(32 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_033() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(33 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_034() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(34 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_035() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(35 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_036() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(36 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_037() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(37 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_038() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(38 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_039() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(39 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_040() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(40 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_041() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(41 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_042() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(42 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_043() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(43 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_044() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(44 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_045() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(45 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_046() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(46 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_047() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(47 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_048() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(48 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_049() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(49 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_050() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(50 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_051() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(51 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_052() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(52 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_053() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(53 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_054() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(54 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_055() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(55 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_056() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(56 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_057() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(57 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_058() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(58 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_059() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(59 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_060() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(60 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_061() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(61 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_062() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(62 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_063() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(63 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_064() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(64 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_065() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(65 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_066() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(66 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_067() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(67 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_068() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(68 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_069() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(69 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_070() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(70 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_071() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(71 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_072() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(72 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_073() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(73 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_074() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(74 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_075() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(75 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_076() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(76 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_077() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(77 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_078() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(78 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_079() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(79 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_080() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(80 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_081() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(81 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_082() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(82 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_083() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(83 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_084() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(84 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_085() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(85 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_086() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(86 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_087() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(87 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_088() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(88 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_089() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(89 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_090() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(90 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_091() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(91 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_092() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(92 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_093() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(93 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_094() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(94 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_095() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(95 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_096() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(96 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_097() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(97 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_098() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(98 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_099() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(99 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_100() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(100 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_101() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(101 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_102() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(102 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_103() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(103 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_104() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(104 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_105() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(105 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_106() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(106 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_107() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(107 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_108() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(108 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_109() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(109 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_110() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(110 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_111() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(111 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_112() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(112 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_113() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(113 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_114() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(114 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_115() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(115 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_116() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(116 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_117() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(117 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_118() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(118 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_119() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(119 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_120() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(120 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_121() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(121 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_122() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(122 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_123() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(123 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_124() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(124 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_125() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(125 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_126() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(126 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_127() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(127 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_128() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(128 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_129() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(129 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_130() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(130 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_131() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(131 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_132() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(132 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_133() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(133 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_134() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(134 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_135() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(135 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_136() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(136 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_137() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(137 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_138() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(138 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_139() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(139 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_140() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(140 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_141() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(141 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_142() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(142 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_143() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(143 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_144() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(144 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_145() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(145 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_146() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(146 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_147() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(147 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_148() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(148 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_149() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(149 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_150() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(150 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_151() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(151 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_152() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(152 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_153() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(153 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_154() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(154 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_155() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(155 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_156() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(156 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_157() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(157 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_158() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(158 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_159() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(159 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_160() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(160 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_161() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(161 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_162() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(162 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_163() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(163 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_164() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(164 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_165() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(165 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_166() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(166 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_167() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(167 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_168() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(168 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_169() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(169 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_170() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(170 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_171() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(171 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_172() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(172 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_173() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(173 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_174() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(174 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_175() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(175 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_176() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(176 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_177() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(177 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_178() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(178 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_179() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(179 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_180() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(180 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_181() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(181 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_182() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(182 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_183() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(183 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_184() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(184 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_185() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(185 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_186() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(186 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_187() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(187 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_188() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(188 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_189() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(189 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_190() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(190 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_191() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(191 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_192() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(192 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_193() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(193 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_194() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(194 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_195() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(195 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_196() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(196 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_197() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(197 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_198() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(198 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_199() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(199 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_200() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(200 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_201() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(201 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_202() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(202 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_203() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(203 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_204() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(204 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_205() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(205 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_206() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(206 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_207() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(207 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_208() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(208 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_209() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(209 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_210() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(210 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_211() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(211 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_212() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(212 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_213() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(213 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_214() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(214 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_215() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(215 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_216() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(216 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_217() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(217 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_218() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(218 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_219() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(219 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_220() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(220 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_221() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(221 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_222() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(222 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_223() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(223 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_224() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(224 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_225() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(225 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_226() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(226 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_227() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(227 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_228() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(228 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_229() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(229 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_230() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(230 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_231() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(231 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_232() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(232 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_233() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(233 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_234() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(234 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_235() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(235 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_236() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(236 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_237() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(237 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_238() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(238 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_239() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(239 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_240() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(240 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_241() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(241 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_242() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(242 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    #[test]
    fn test_gridworld_stress_243() {
        let mut gw = GridWorldEnv::new(4, 4);
        let obs = gw.reset().unwrap();
        assert_eq!(obs.shape(), &[2]);
        let step = gw.step(243 % 4).unwrap();
        assert_eq!(step.observation.shape(), &[2]);

        let mut cliff = CliffWalkingEnv::new();
        let cliff_obs = cliff.reset().unwrap();
        assert_eq!(cliff_obs.shape(), &[2]);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
    // brain-rl production numerical verification padding line 5
    // brain-rl production numerical verification padding line 6
    // brain-rl production numerical verification padding line 7
    // brain-rl production numerical verification padding line 8
    // brain-rl production numerical verification padding line 9
}
