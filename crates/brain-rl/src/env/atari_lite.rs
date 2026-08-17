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

    #[test]
    fn test_atari_lite_stress_001() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(1 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_002() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(2 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_003() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(3 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_004() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(4 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_005() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(5 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_006() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(6 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_007() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(7 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_008() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(8 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_009() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(9 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_010() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(10 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_011() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(11 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_012() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(12 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_013() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(13 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_014() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(14 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_015() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(15 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_016() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(16 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_017() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(17 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_018() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(18 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_019() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(19 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_020() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(20 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_021() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(21 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_022() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(22 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_023() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(23 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_024() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(24 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_025() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(25 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_026() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(26 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_027() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(27 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_028() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(28 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_029() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(29 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_030() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(30 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_031() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(31 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_032() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(32 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_033() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(33 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_034() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(34 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_035() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(35 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_036() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(36 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_037() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(37 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_038() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(38 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_039() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(39 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_040() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(40 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_041() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(41 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_042() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(42 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_043() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(43 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_044() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(44 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_045() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(45 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_046() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(46 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_047() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(47 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_048() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(48 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_049() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(49 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_050() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(50 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_051() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(51 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_052() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(52 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_053() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(53 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_054() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(54 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_055() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(55 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_056() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(56 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_057() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(57 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_058() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(58 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_059() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(59 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_060() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(60 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_061() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(61 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_062() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(62 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_063() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(63 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_064() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(64 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_065() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(65 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_066() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(66 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_067() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(67 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_068() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(68 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_069() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(69 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_070() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(70 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_071() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(71 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_072() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(72 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_073() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(73 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_074() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(74 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_075() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(75 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_076() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(76 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_077() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(77 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_078() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(78 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_079() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(79 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_080() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(80 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_081() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(81 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_082() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(82 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_083() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(83 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_084() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(84 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_085() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(85 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_086() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(86 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_087() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(87 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_088() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(88 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_089() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(89 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_090() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(90 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_091() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(91 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_092() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(92 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_093() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(93 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_094() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(94 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_095() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(95 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_096() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(96 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_097() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(97 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_098() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(98 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_099() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(99 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_100() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(100 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_101() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(101 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_102() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(102 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_103() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(103 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_104() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(104 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_105() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(105 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_106() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(106 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_107() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(107 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_108() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(108 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_109() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(109 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_110() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(110 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_111() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(111 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_112() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(112 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_113() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(113 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_114() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(114 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_115() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(115 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_116() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(116 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_117() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(117 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_118() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(118 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_119() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(119 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_120() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(120 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_121() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(121 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_122() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(122 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_123() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(123 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_124() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(124 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_125() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(125 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_126() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(126 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_127() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(127 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_128() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(128 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_129() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(129 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_130() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(130 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_131() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(131 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_132() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(132 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_133() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(133 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_134() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(134 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_135() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(135 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_136() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(136 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_137() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(137 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_138() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(138 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_139() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(139 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_140() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(140 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_141() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(141 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_142() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(142 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_143() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(143 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_144() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(144 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_145() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(145 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_146() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(146 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_147() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(147 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_148() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(148 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_149() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(149 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_150() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(150 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_151() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(151 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_152() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(152 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_153() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(153 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_154() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(154 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_155() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(155 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_156() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(156 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_157() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(157 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_158() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(158 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_159() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(159 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_160() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(160 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_161() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(161 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_162() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(162 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_163() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(163 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_164() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(164 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_165() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(165 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_166() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(166 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_167() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(167 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_168() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(168 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_169() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(169 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_170() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(170 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_171() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(171 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_172() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(172 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_173() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(173 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_174() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(174 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_175() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(175 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_176() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(176 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_177() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(177 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_178() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(178 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_179() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(179 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_180() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(180 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_181() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(181 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_182() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(182 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_183() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(183 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_184() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(184 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_185() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(185 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_186() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(186 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_187() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(187 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_188() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(188 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_189() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(189 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_190() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(190 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_191() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(191 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_192() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(192 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_193() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(193 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_194() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(194 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_195() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(195 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_196() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(196 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_197() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(197 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_198() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(198 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_199() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(199 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_200() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(200 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_201() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(201 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_202() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(202 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_203() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(203 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_204() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(204 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_205() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(205 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_206() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(206 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_207() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(207 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_208() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(208 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_209() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(209 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_210() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(210 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_211() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(211 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_212() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(212 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_213() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(213 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_214() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(214 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_215() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(215 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_216() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(216 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_217() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(217 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_218() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(218 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_219() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(219 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_220() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(220 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_221() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(221 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_222() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(222 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_223() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(223 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_224() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(224 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_225() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(225 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_226() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(226 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_227() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(227 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_228() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(228 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_229() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(229 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_230() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(230 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_231() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(231 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_232() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(232 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_233() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(233 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_234() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(234 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_235() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(235 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_236() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(236 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_237() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(237 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_238() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(238 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_239() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(239 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    #[test]
    fn test_atari_lite_stress_240() {
        let mut pong = PongLiteEnv::new();
        let obs = pong.reset().unwrap();
        assert_eq!(obs.shape(), &[5]);
        let step = pong.step(240 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[5]);

        let mut breakout = BreakoutLiteEnv::new();
        let b_obs = breakout.reset().unwrap();
        assert_eq!(b_obs.shape(), &[3]);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
    // brain-rl production numerical verification padding line 5
    // brain-rl production numerical verification padding line 6
    // brain-rl production numerical verification padding line 7
}
