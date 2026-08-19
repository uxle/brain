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
}
