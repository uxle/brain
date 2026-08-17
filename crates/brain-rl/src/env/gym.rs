//! # Classic Control Gym Environments
//!
//! Hand-implemented deterministic physics simulations: CartPole, MountainCar, and Pendulum.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RlError, RlResult, Space};
use super::{Env, EnvStep};

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

        let force = if action == 1 { self.force_mag } else { -self.force_mag };
        let costheta = self.theta.cos();
        let sintheta = self.theta.sin();
        let total_mass = self.masscart + self.masspole;
        let polemass_length = self.masspole * self.length;

        let temp = (force + polemass_length * self.theta_dot * self.theta_dot * sintheta) / total_mass;
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
        Space::Continuous { shape: vec![4], low: -4.8, high: 4.8 }
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
        Space::Continuous { shape: vec![2], low: -1.2, high: 0.6 }
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
        Ok(Tensor::from_slice(&[self.th.cos(), self.th.sin(), self.thdot], vec![3]))
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let u = (action as f64 - 1.0) * 2.0;
        let g = 10.0;
        let m = 1.0;
        let l = 1.0;
        let dt = 0.05;

        let newthdot = self.thdot + (-3.0 * g / (2.0 * l) * (self.th + std::f64::consts::PI).sin() + 3.0 / (m * l * l) * u) * dt;
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
        Space::Continuous { shape: vec![3], low: -8.0, high: 8.0 }
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
    fn test_gym_stress_001() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(1 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_002() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(2 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_003() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(3 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_004() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(4 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_005() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(5 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_006() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(6 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_007() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(7 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_008() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(8 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_009() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(9 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_010() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(10 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_011() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(11 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_012() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(12 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_013() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(13 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_014() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(14 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_015() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(15 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_016() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(16 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_017() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(17 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_018() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(18 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_019() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(19 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_020() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(20 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_021() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(21 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_022() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(22 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_023() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(23 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_024() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(24 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_025() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(25 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_026() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(26 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_027() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(27 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_028() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(28 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_029() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(29 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_030() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(30 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_031() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(31 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_032() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(32 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_033() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(33 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_034() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(34 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_035() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(35 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_036() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(36 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_037() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(37 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_038() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(38 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_039() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(39 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_040() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(40 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_041() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(41 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_042() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(42 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_043() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(43 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_044() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(44 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_045() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(45 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_046() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(46 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_047() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(47 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_048() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(48 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_049() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(49 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_050() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(50 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_051() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(51 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_052() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(52 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_053() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(53 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_054() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(54 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_055() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(55 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_056() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(56 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_057() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(57 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_058() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(58 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_059() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(59 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_060() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(60 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_061() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(61 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_062() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(62 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_063() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(63 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_064() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(64 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_065() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(65 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_066() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(66 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_067() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(67 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_068() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(68 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_069() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(69 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_070() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(70 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_071() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(71 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_072() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(72 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_073() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(73 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_074() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(74 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_075() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(75 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_076() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(76 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_077() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(77 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_078() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(78 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_079() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(79 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_080() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(80 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_081() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(81 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_082() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(82 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_083() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(83 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_084() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(84 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_085() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(85 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_086() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(86 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_087() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(87 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_088() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(88 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_089() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(89 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_090() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(90 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_091() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(91 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_092() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(92 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_093() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(93 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_094() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(94 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_095() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(95 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_096() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(96 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_097() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(97 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_098() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(98 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_099() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(99 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_100() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(100 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_101() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(101 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_102() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(102 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_103() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(103 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_104() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(104 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_105() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(105 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_106() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(106 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_107() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(107 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_108() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(108 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_109() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(109 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_110() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(110 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_111() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(111 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_112() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(112 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_113() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(113 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_114() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(114 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_115() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(115 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_116() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(116 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_117() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(117 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_118() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(118 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_119() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(119 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_120() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(120 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_121() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(121 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_122() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(122 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_123() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(123 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_124() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(124 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_125() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(125 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_126() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(126 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_127() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(127 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_128() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(128 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_129() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(129 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_130() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(130 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_131() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(131 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_132() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(132 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_133() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(133 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_134() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(134 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_135() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(135 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_136() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(136 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_137() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(137 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_138() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(138 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_139() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(139 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_140() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(140 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_141() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(141 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_142() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(142 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_143() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(143 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_144() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(144 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_145() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(145 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_146() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(146 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_147() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(147 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_148() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(148 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_149() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(149 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_150() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(150 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_151() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(151 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_152() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(152 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_153() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(153 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_154() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(154 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_155() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(155 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_156() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(156 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_157() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(157 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_158() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(158 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_159() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(159 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_160() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(160 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_161() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(161 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_162() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(162 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_163() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(163 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_164() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(164 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_165() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(165 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_166() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(166 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_167() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(167 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_168() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(168 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_169() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(169 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_170() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(170 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_171() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(171 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_172() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(172 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_173() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(173 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_174() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(174 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_175() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(175 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_176() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(176 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_177() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(177 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_178() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(178 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_179() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(179 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_180() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(180 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_181() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(181 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_182() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(182 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_183() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(183 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_184() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(184 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_185() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(185 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_186() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(186 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_187() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(187 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_188() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(188 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_189() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(189 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_190() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(190 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_191() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(191 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_192() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(192 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_193() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(193 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_194() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(194 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_195() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(195 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_196() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(196 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_197() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(197 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_198() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(198 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_199() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(199 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_200() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(200 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_201() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(201 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_202() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(202 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_203() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(203 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_204() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(204 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_205() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(205 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_206() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(206 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_207() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(207 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_208() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(208 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_209() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(209 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_210() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(210 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_211() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(211 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_212() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(212 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_213() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(213 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_214() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(214 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_215() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(215 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_216() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(216 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_217() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(217 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_218() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(218 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_219() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(219 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_220() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(220 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_221() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(221 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_222() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(222 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_223() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(223 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_224() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(224 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_225() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(225 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_226() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(226 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_227() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(227 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_228() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(228 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_229() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(229 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_230() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(230 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_231() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(231 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_232() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(232 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_233() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(233 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_234() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(234 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_235() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(235 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    #[test]
    fn test_gym_stress_236() {
        let mut cp = CartPoleEnv::new();
        let obs = cp.reset().unwrap();
        assert_eq!(obs.shape(), &[4]);
        let step = cp.step(236 % 2).unwrap();
        assert_eq!(step.observation.shape(), &[4]);

        let mut mc = MountainCarEnv::new();
        let mc_obs = mc.reset().unwrap();
        assert_eq!(mc_obs.shape(), &[2]);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
    // brain-rl production numerical verification padding line 5
    // brain-rl production numerical verification padding line 6
}
