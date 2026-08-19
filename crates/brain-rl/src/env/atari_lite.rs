//! # Atari-Lite Feature & Pixel Environments
//!
//! Compact, pure-Rust Atari-style simulation environments: PongLite and BreakoutLite.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RlError, RlResult, Space};
use super::{Env, EnvStep};

/// Pong-Lite 2D paddle and ball simulation.
#[derive(Debug, Clone)]
pub struct PongLiteEnv {
    pub paddle_y: f64,
    pub ball_x: f64,
    pub ball_y: f64,
    pub ball_vx: f64,
    pub ball_vy: f64,
    pub step_count: usize,
    pub max_steps: usize,
}

impl Default for PongLiteEnv {
    fn default() -> Self {
        Self {
            paddle_y: 0.5,
            ball_x: 0.5,
            ball_y: 0.5,
            ball_vx: 0.05,
            ball_vy: 0.03,
            step_count: 0,
            max_steps: 1000,
        }
    }
}

impl PongLiteEnv {
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> Tensor {
        Tensor::from_slice(&[self.paddle_y, self.ball_x, self.ball_y, self.ball_vx, self.ball_vy], vec![5])
    }
}

impl Env for PongLiteEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.paddle_y = 0.5;
        self.ball_x = 0.5;
        self.ball_y = 0.5;
        self.ball_vx = 0.05;
        self.ball_vy = 0.03;
        self.step_count = 0;
        Ok(self.state())
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        match action {
            0 => {},
            1 => self.paddle_y = (self.paddle_y - 0.05).max(0.0),
            2 => self.paddle_y = (self.paddle_y + 0.05).min(1.0),
            _ => return Err(RlError::InvalidAction(action)),
        }

        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        if self.ball_y <= 0.0 || self.ball_y >= 1.0 {
            self.ball_vy = -self.ball_vy;
        }

        let mut reward = 0.0;
        let mut done = false;

        if self.ball_x >= 1.0 {
            self.ball_vx = -self.ball_vx;
        }

        if self.ball_x <= 0.0 {
            if (self.ball_y - self.paddle_y).abs() < 0.15 {
                reward = 1.0;
                self.ball_vx = -self.ball_vx;
            } else {
                reward = -1.0;
                done = true;
            }
        }

        self.step_count += 1;
        let truncated = self.step_count >= self.max_steps;

        Ok(EnvStep::new(self.state(), reward, done, truncated))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous { shape: vec![5], low: 0.0, high: 1.0 }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }
}

/// Breakout-Lite paddle and brick environment.
#[derive(Debug, Clone)]
pub struct BreakoutLiteEnv {
    pub paddle_x: f64,
    pub ball_x: f64,
    pub ball_y: f64,
    pub ball_vx: f64,
    pub ball_vy: f64,
    pub bricks_left: usize,
    pub step_count: usize,
}

impl Default for BreakoutLiteEnv {
    fn default() -> Self {
        Self {
            paddle_x: 0.5,
            ball_x: 0.5,
            ball_y: 0.2,
            ball_vx: 0.04,
            ball_vy: 0.04,
            bricks_left: 20,
            step_count: 0,
        }
    }
}

impl BreakoutLiteEnv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Env for BreakoutLiteEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.paddle_x = 0.5;
        self.ball_x = 0.5;
        self.ball_y = 0.2;
        self.ball_vx = 0.04;
        self.ball_vy = 0.04;
        self.bricks_left = 20;
        self.step_count = 0;
        Ok(Tensor::from_slice(&[self.paddle_x, self.ball_x, self.ball_y], vec![3]))
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        match action {
            0 => {},
            1 => self.paddle_x = (self.paddle_x - 0.05).max(0.0),
            2 => self.paddle_x = (self.paddle_x + 0.05).min(1.0),
            _ => return Err(RlError::InvalidAction(action)),
        }

        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        if self.ball_x <= 0.0 || self.ball_x >= 1.0 {
            self.ball_vx = -self.ball_vx;
        }
        if self.ball_y >= 1.0 {
            self.ball_vy = -self.ball_vy;
        }

        let mut reward = 0.0;
        let mut done = false;

        if self.ball_y <= 0.0 {
            if (self.ball_x - self.paddle_x).abs() < 0.15 {
                self.ball_vy = -self.ball_vy;
                reward = 0.1;
            } else {
                reward = -1.0;
                done = true;
            }
        }

        self.step_count += 1;
        let truncated = self.step_count >= 500;

        Ok(EnvStep::new(
            Tensor::from_slice(&[self.paddle_x, self.ball_x, self.ball_y], vec![3]),
            reward,
            done,
            truncated,
        ))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous { shape: vec![3], low: 0.0, high: 1.0 }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
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
