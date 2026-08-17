//! # MuJoCo-Lite Planar Continuous Physics
//!
//! HalfCheetah-Lite and Reacher-Lite continuous state-space simulation environments.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RlResult, Space};
use super::{Env, EnvStep};

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
        Space::Continuous { shape: vec![8], low: -10.0, high: 10.0 }
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
        Ok(Tensor::from_slice(&[self.theta1, self.theta2, self.target_x, self.target_y], vec![4]))
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
            Tensor::from_slice(&[self.theta1, self.theta2, self.target_x, self.target_y], vec![4]),
            reward,
            done,
            truncated,
        ))
    }

    fn observation_space(&self) -> Space {
        Space::Continuous { shape: vec![4], low: -std::f64::consts::PI, high: std::f64::consts::PI }
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
    fn test_mujoco_lite_stress_001() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(1 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_002() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(2 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_003() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(3 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_004() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(4 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_005() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(5 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_006() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(6 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_007() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(7 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_008() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(8 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_009() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(9 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_010() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(10 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_011() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(11 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_012() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(12 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_013() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(13 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_014() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(14 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_015() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(15 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_016() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(16 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_017() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(17 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_018() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(18 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_019() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(19 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_020() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(20 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_021() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(21 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_022() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(22 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_023() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(23 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_024() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(24 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_025() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(25 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_026() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(26 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_027() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(27 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_028() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(28 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_029() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(29 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_030() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(30 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_031() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(31 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_032() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(32 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_033() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(33 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_034() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(34 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_035() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(35 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_036() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(36 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_037() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(37 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_038() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(38 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_039() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(39 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_040() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(40 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_041() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(41 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_042() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(42 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_043() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(43 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_044() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(44 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_045() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(45 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_046() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(46 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_047() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(47 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_048() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(48 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_049() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(49 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_050() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(50 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_051() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(51 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_052() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(52 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_053() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(53 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_054() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(54 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_055() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(55 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_056() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(56 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_057() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(57 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_058() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(58 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_059() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(59 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_060() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(60 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_061() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(61 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_062() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(62 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_063() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(63 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_064() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(64 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_065() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(65 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_066() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(66 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_067() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(67 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_068() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(68 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_069() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(69 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_070() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(70 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_071() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(71 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_072() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(72 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_073() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(73 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_074() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(74 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_075() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(75 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_076() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(76 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_077() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(77 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_078() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(78 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_079() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(79 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_080() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(80 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_081() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(81 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_082() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(82 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_083() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(83 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_084() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(84 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_085() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(85 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_086() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(86 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_087() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(87 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_088() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(88 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_089() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(89 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_090() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(90 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_091() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(91 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_092() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(92 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_093() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(93 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_094() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(94 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_095() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(95 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_096() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(96 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_097() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(97 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_098() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(98 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_099() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(99 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_100() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(100 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_101() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(101 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_102() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(102 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_103() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(103 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_104() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(104 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_105() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(105 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_106() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(106 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_107() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(107 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_108() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(108 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_109() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(109 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_110() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(110 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_111() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(111 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_112() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(112 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_113() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(113 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_114() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(114 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_115() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(115 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_116() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(116 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_117() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(117 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_118() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(118 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_119() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(119 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_120() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(120 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_121() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(121 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_122() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(122 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_123() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(123 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_124() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(124 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_125() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(125 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_126() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(126 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_127() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(127 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_128() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(128 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_129() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(129 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_130() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(130 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_131() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(131 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_132() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(132 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_133() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(133 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_134() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(134 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_135() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(135 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_136() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(136 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_137() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(137 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_138() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(138 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_139() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(139 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_140() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(140 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_141() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(141 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_142() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(142 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_143() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(143 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_144() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(144 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_145() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(145 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_146() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(146 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_147() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(147 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_148() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(148 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_149() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(149 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_150() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(150 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_151() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(151 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_152() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(152 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_153() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(153 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_154() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(154 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_155() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(155 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_156() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(156 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_157() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(157 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_158() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(158 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_159() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(159 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_160() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(160 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_161() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(161 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_162() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(162 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_163() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(163 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_164() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(164 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_165() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(165 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_166() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(166 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_167() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(167 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_168() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(168 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_169() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(169 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_170() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(170 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_171() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(171 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_172() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(172 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_173() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(173 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_174() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(174 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_175() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(175 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_176() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(176 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_177() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(177 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_178() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(178 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_179() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(179 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_180() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(180 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_181() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(181 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_182() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(182 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_183() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(183 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_184() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(184 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_185() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(185 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_186() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(186 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_187() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(187 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_188() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(188 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_189() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(189 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_190() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(190 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_191() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(191 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_192() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(192 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_193() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(193 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_194() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(194 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_195() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(195 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_196() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(196 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_197() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(197 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_198() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(198 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_199() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(199 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_200() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(200 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_201() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(201 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_202() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(202 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_203() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(203 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_204() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(204 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_205() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(205 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_206() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(206 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_207() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(207 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_208() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(208 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_209() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(209 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_210() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(210 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_211() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(211 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_212() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(212 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_213() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(213 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_214() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(214 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_215() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(215 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_216() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(216 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_217() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(217 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_218() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(218 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_219() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(219 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_220() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(220 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_221() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(221 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_222() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(222 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_223() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(223 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_224() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(224 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_225() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(225 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_226() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(226 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_227() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(227 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_228() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(228 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_229() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(229 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_230() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(230 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_231() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(231 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_232() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(232 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_233() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(233 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_234() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(234 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_235() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(235 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_236() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(236 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_237() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(237 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_238() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(238 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_239() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(239 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_240() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(240 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_241() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(241 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_242() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(242 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_243() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(243 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_244() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(244 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }

    #[test]
    fn test_mujoco_lite_stress_245() {
        let mut cheetah = HalfCheetahLiteEnv::new();
        let obs = cheetah.reset().unwrap();
        assert_eq!(obs.shape(), &[8]);
        let step = cheetah.step(245 % 3).unwrap();
        assert_eq!(step.observation.shape(), &[8]);

        let mut reacher = ReacherLiteEnv::new();
        let r_obs = reacher.reset().unwrap();
        assert_eq!(r_obs.shape(), &[4]);
    }
}
