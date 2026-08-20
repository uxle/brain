//! # MuJoCo-Lite Planar Continuous Physics
//!
//! HalfCheetah-Lite and Reacher-Lite continuous state-space simulation environments.
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

use super::super::core::{RlResult, Space};
use super::{Env, EnvStep};
use brain_core::Tensor;

/// Planar HalfCheetah locomotion environment.
#[derive(Debug, Clone)]
pub struct HalfCheetahLiteEnv {
    pub x_pos: f64,
    pub x_vel: f64,
    pub joint_angles: Vec<f64>,
    pub step_count: usize,
    pub max_steps: usize,
}

impl Default for HalfCheetahLiteEnv {
    fn default() -> Self {
        Self {
            x_pos: 0.0,
            x_vel: 0.0,
            joint_angles: vec![0.0; 6],
            step_count: 0,
            max_steps: 1000,
        }
    }
}

impl HalfCheetahLiteEnv {
    pub fn new() -> Self {
        Self::default()
    }

    fn state(&self) -> Tensor {
        let mut s = vec![self.x_pos, self.x_vel];
        s.extend_from_slice(&self.joint_angles);
        Tensor::from_slice(&s, vec![8])
    }
}

impl Env for HalfCheetahLiteEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.x_pos = 0.0;
        self.x_vel = 0.0;
        self.joint_angles = vec![0.0; 6];
        self.step_count = 0;
        Ok(self.state())
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let torque = (action as f64 - 1.0) * 0.5;
        self.x_vel += torque * 0.1;
        self.x_vel = self.x_vel.clamp(-2.0, 5.0);
        self.x_pos += self.x_vel * 0.02;

        self.step_count += 1;
        let reward = self.x_vel - 0.1 * (torque * torque);
        let done = false;
        let truncated = self.step_count >= self.max_steps;

        Ok(EnvStep::new(self.state(), reward, done, truncated))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous {
            shape: vec![8],
            low: -10.0,
            high: 10.0,
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }
}

/// 2-DOF Planar Arm Reacher Environment.
#[derive(Debug, Clone)]
pub struct ReacherLiteEnv {
    pub theta1: f64,
    pub theta2: f64,
    pub target_x: f64,
    pub target_y: f64,
    pub step_count: usize,
}

impl Default for ReacherLiteEnv {
    fn default() -> Self {
        Self {
            theta1: 0.0,
            theta2: 0.0,
            target_x: 0.5,
            target_y: 0.5,
            step_count: 0,
        }
    }
}

impl ReacherLiteEnv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Env for ReacherLiteEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.theta1 = 0.0;
        self.theta2 = 0.0;
        self.step_count = 0;
        Ok(Tensor::from_slice(
            &[self.theta1, self.theta2, self.target_x, self.target_y],
            vec![4],
        ))
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let delta = (action as f64 - 1.0) * 0.1;
        self.theta1 += delta;
        self.step_count += 1;

        let end_x = self.theta1.cos() + self.theta2.cos();
        let end_y = self.theta1.sin() + self.theta2.sin();

        let dist_sq = (end_x - self.target_x).powi(2) + (end_y - self.target_y).powi(2);
        let reward = -dist_sq;
        let done = dist_sq < 0.05;
        let truncated = self.step_count >= 200;

        Ok(EnvStep::new(
            Tensor::from_slice(
                &[self.theta1, self.theta2, self.target_x, self.target_y],
                vec![4],
            ),
            reward,
            done,
            truncated,
        ))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous {
            shape: vec![4],
            low: -std::f64::consts::PI,
            high: std::f64::consts::PI,
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }
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
