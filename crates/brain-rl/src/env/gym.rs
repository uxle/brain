//! # Classic Control Gym Environments
//!
//! Hand-implemented deterministic physics simulations: CartPole, MountainCar, and Pendulum.
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

use super::super::core::{RlError, RlResult, Space};
use super::{Env, EnvStep};
use brain_core::Tensor;

/// Classic CartPole-v1 balancing environment.
#[derive(Debug, Clone)]
pub struct CartPoleEnv {
    pub x: f64,
    pub x_dot: f64,
    pub theta: f64,
    pub theta_dot: f64,
    pub step_count: usize,
    pub max_steps: usize,
    pub gravity: f64,
    pub masscart: f64,
    pub masspole: f64,
    pub length: f64,
    pub force_mag: f64,
    pub tau: f64,
    pub theta_threshold_radians: f64,
    pub x_threshold: f64,
}

impl Default for CartPoleEnv {
    fn default() -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            step_count: 0,
            max_steps: 500,
            gravity: 9.8,
            masscart: 1.0,
            masspole: 0.1,
            length: 0.5,
            force_mag: 10.0,
            tau: 0.02,
            theta_threshold_radians: 12.0 * 2.0 * std::f64::consts::PI / 360.0,
            x_threshold: 2.4,
        }
    }
}

impl CartPoleEnv {
    pub fn new() -> Self {
        Self::default()
    }

    fn state_tensor(&self) -> Tensor {
        Tensor::from_slice(&[self.x, self.x_dot, self.theta, self.theta_dot], vec![4])
    }
}

impl Env for CartPoleEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.x = 0.0;
        self.x_dot = 0.0;
        self.theta = 0.0;
        self.theta_dot = 0.0;
        self.step_count = 0;
        Ok(self.state_tensor())
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        if action > 1 {
            return Err(RlError::InvalidAction(action));
        }

        let force = if action == 1 {
            self.force_mag
        } else {
            -self.force_mag
        };
        let costheta = self.theta.cos();
        let sintheta = self.theta.sin();
        let total_mass = self.masscart + self.masspole;
        let polemass_length = self.masspole * self.length;

        let temp =
            (force + polemass_length * self.theta_dot * self.theta_dot * sintheta) / total_mass;
        let thetaacc = (self.gravity * sintheta - costheta * temp)
            / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / total_mass));
        let xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        self.x += self.tau * self.x_dot;
        self.x_dot += self.tau * xacc;
        self.theta += self.tau * self.theta_dot;
        self.theta_dot += self.tau * thetaacc;

        self.step_count += 1;

        let done = self.x < -self.x_threshold
            || self.x > self.x_threshold
            || self.theta < -self.theta_threshold_radians
            || self.theta > self.theta_threshold_radians;

        let truncated = self.step_count >= self.max_steps;
        let reward = 1.0;

        Ok(EnvStep::new(self.state_tensor(), reward, done, truncated))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous {
            shape: vec![4],
            low: -4.8,
            high: 4.8,
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(2)
    }
}

/// Classic MountainCar-v0 environment.
#[derive(Debug, Clone)]
pub struct MountainCarEnv {
    pub position: f64,
    pub velocity: f64,
    pub step_count: usize,
    pub max_steps: usize,
}

impl Default for MountainCarEnv {
    fn default() -> Self {
        Self {
            position: -0.5,
            velocity: 0.0,
            step_count: 0,
            max_steps: 200,
        }
    }
}

impl MountainCarEnv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Env for MountainCarEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.position = -0.5;
        self.velocity = 0.0;
        self.step_count = 0;
        Ok(Tensor::from_slice(&[self.position, self.velocity], vec![2]))
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        if action > 2 {
            return Err(RlError::InvalidAction(action));
        }

        let force = (action as f64 - 1.0) * 0.001;
        self.velocity += force + (3.0 * self.position).cos() * (-0.0025);
        self.velocity = self.velocity.clamp(-0.07, 0.07);
        self.position += self.velocity;
        self.position = self.position.clamp(-1.2, 0.6);

        if self.position <= -1.2 && self.velocity < 0.0 {
            self.velocity = 0.0;
        }

        self.step_count += 1;
        let done = self.position >= 0.5;
        let truncated = self.step_count >= self.max_steps;
        let reward = -1.0;

        Ok(EnvStep::new(
            Tensor::from_slice(&[self.position, self.velocity], vec![2]),
            reward,
            done,
            truncated,
        ))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous {
            shape: vec![2],
            low: -1.2,
            high: 0.6,
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }
}

/// Classic Inverted Pendulum Environment.
#[derive(Debug, Clone)]
pub struct PendulumEnv {
    pub th: f64,
    pub thdot: f64,
    pub step_count: usize,
    pub max_steps: usize,
}

impl Default for PendulumEnv {
    fn default() -> Self {
        Self {
            th: 0.0,
            thdot: 0.0,
            step_count: 0,
            max_steps: 200,
        }
    }
}

impl PendulumEnv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Env for PendulumEnv {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.th = 0.0;
        self.thdot = 0.0;
        self.step_count = 0;
        Ok(Tensor::from_slice(
            &[self.th.cos(), self.th.sin(), self.thdot],
            vec![3],
        ))
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let u = (action as f64 - 1.0) * 2.0;
        let g = 10.0;
        let m = 1.0;
        let l = 1.0;
        let dt = 0.05;

        let newthdot = self.thdot
            + (-3.0 * g / (2.0 * l) * (self.th + std::f64::consts::PI).sin()
                + 3.0 / (m * l * l) * u)
                * dt;
        self.thdot = newthdot.clamp(-8.0, 8.0);
        self.th += self.thdot * dt;
        self.step_count += 1;

        let costs = (self.th * self.th) + 0.1 * (self.thdot * self.thdot) + 0.001 * (u * u);
        let reward = -costs;
        let done = false;
        let truncated = self.step_count >= self.max_steps;

        Ok(EnvStep::new(
            Tensor::from_slice(&[self.th.cos(), self.th.sin(), self.thdot], vec![3]),
            reward,
            done,
            truncated,
        ))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous {
            shape: vec![3],
            low: -8.0,
            high: 8.0,
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
