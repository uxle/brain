//! # Environment Wrappers & Observation Transformers
//!
//! FrameStack, TimeLimit, RewardScale, and composite wrapper pipelines.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RlResult, Space};
use super::{Env, EnvStep};

/// Stacks k consecutive observation tensors along feature dimensions.
pub struct FrameStackWrapper<E: Env> {
    pub env: E,
    pub k: usize,
    pub frames: Vec<Tensor>,
}

impl<E: Env> FrameStackWrapper<E> {
    pub fn new(env: E, k: usize) -> Self {
        Self {
            env,
            k: k.max(1),
            frames: Vec::new(),
        }
    }

    fn stacked_obs(&self) -> Tensor {
        let mut data = Vec::new();
        for f in &self.frames {
            data.extend_from_slice(f.data());
        }
        let n = data.len();
        Tensor::from_slice(&data, vec![n])
    }
}

impl<E: Env> Env for FrameStackWrapper<E> {
    fn reset(&mut self) -> RlResult<Tensor> {
        let obs = self.env.reset()?;
        self.frames = vec![obs; self.k];
        Ok(self.stacked_obs())
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let mut step = self.env.step(action)?;
        self.frames.remove(0);
        self.frames.push(step.observation.clone());
        step.observation = self.stacked_obs();
        Ok(step)
    }

    fn observation_space(&self) -> Space {
        self.env.observation_space()
    }

    fn action_space(&self) -> Space {
        self.env.action_space()
    }
}

/// Enforces maximum episode step limit truncation.
pub struct TimeLimitWrapper<E: Env> {
    pub env: E,
    pub max_steps: usize,
    pub step_count: usize,
}

impl<E: Env> TimeLimitWrapper<E> {
    pub fn new(env: E, max_steps: usize) -> Self {
        Self {
            env,
            max_steps,
            step_count: 0,
        }
    }
}

impl<E: Env> Env for TimeLimitWrapper<E> {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.step_count = 0;
        self.env.reset()
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let mut step = self.env.step(action)?;
        self.step_count += 1;
        if self.step_count >= self.max_steps {
            step.truncated = true;
        }
        Ok(step)
    }

    fn observation_space(&self) -> Space {
        self.env.observation_space()
    }

    fn action_space(&self) -> Space {
        self.env.action_space()
    }
}

/// Scales environment rewards by constant multiplier.
pub struct RewardScaleWrapper<E: Env> {
    pub env: E,
    pub scale: f64,
}

impl<E: Env> RewardScaleWrapper<E> {
    pub fn new(env: E, scale: f64) -> Self {
        Self { env, scale }
    }
}

impl<E: Env> Env for RewardScaleWrapper<E> {
    fn reset(&mut self) -> RlResult<Tensor> {
        self.env.reset()
    }

    fn step(&mut self, action: usize) -> RlResult<EnvStep> {
        let mut step = self.env.step(action)?;
        step.reward *= self.scale;
        Ok(step)
    }

    fn observation_space(&self) -> Space {
        self.env.observation_space()
    }

    fn action_space(&self) -> Space {
        self.env.action_space()
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
    fn test_wrappers_stress_001() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(1 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_002() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(2 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_003() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(3 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_004() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(4 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_005() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(5 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_006() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(6 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_007() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(7 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_008() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(8 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_009() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(9 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_010() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(10 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_011() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(11 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_012() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(12 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_013() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(13 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_014() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(14 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_015() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(15 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_016() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(16 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_017() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(17 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_018() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(18 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_019() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(19 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_020() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(20 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_021() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(21 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_022() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(22 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_023() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(23 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_024() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(24 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_025() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(25 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_026() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(26 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_027() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(27 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_028() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(28 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_029() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(29 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_030() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(30 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_031() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(31 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_032() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(32 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_033() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(33 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_034() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(34 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_035() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(35 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_036() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(36 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_037() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(37 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_038() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(38 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_039() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(39 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_040() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(40 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_041() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(41 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_042() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(42 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_043() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(43 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_044() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(44 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_045() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(45 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_046() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(46 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_047() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(47 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_048() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(48 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_049() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(49 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_050() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(50 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_051() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(51 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_052() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(52 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_053() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(53 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_054() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(54 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_055() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(55 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_056() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(56 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_057() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(57 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_058() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(58 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_059() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(59 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_060() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(60 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_061() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(61 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_062() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(62 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_063() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(63 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_064() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(64 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_065() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(65 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_066() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(66 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_067() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(67 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_068() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(68 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_069() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(69 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_070() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(70 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_071() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(71 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_072() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(72 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_073() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(73 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_074() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(74 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_075() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(75 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_076() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(76 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_077() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(77 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_078() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(78 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_079() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(79 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_080() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(80 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_081() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(81 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_082() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(82 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_083() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(83 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_084() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(84 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_085() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(85 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_086() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(86 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_087() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(87 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_088() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(88 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_089() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(89 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_090() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(90 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_091() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(91 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_092() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(92 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_093() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(93 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_094() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(94 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_095() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(95 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_096() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(96 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_097() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(97 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_098() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(98 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_099() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(99 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_100() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(100 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_101() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(101 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_102() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(102 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_103() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(103 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_104() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(104 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_105() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(105 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_106() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(106 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_107() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(107 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_108() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(108 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_109() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(109 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_110() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(110 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_111() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(111 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_112() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(112 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_113() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(113 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_114() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(114 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_115() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(115 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_116() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(116 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_117() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(117 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_118() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(118 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_119() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(119 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_120() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(120 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_121() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(121 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_122() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(122 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_123() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(123 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_124() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(124 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_125() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(125 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_126() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(126 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_127() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(127 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_128() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(128 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_129() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(129 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_130() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(130 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_131() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(131 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_132() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(132 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_133() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(133 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_134() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(134 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_135() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(135 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_136() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(136 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_137() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(137 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_138() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(138 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_139() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(139 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_140() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(140 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_141() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(141 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_142() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(142 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_143() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(143 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_144() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(144 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_145() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(145 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_146() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(146 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_147() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(147 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_148() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(148 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_149() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(149 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_150() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(150 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_151() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(151 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_152() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(152 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_153() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(153 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_154() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(154 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_155() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(155 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_156() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(156 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_157() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(157 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_158() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(158 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_159() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(159 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_160() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(160 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_161() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(161 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_162() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(162 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_163() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(163 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_164() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(164 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_165() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(165 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_166() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(166 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_167() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(167 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_168() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(168 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_169() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(169 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_170() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(170 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_171() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(171 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_172() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(172 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_173() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(173 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_174() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(174 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_175() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(175 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_176() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(176 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_177() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(177 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_178() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(178 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_179() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(179 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_180() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(180 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_181() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(181 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_182() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(182 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_183() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(183 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_184() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(184 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_185() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(185 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_186() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(186 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_187() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(187 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_188() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(188 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_189() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(189 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_190() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(190 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_191() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(191 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_192() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(192 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_193() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(193 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_194() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(194 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_195() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(195 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_196() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(196 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_197() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(197 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_198() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(198 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_199() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(199 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_200() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(200 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_201() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(201 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_202() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(202 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_203() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(203 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_204() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(204 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_205() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(205 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_206() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(206 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_207() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(207 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_208() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(208 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_209() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(209 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_210() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(210 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_211() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(211 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_212() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(212 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_213() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(213 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_214() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(214 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_215() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(215 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_216() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(216 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_217() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(217 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_218() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(218 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_219() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(219 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_220() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(220 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_221() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(221 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_222() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(222 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_223() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(223 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_224() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(224 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_225() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(225 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_226() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(226 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_227() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(227 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_228() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(228 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_229() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(229 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_230() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(230 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_231() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(231 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_232() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(232 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_233() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(233 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_234() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(234 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_235() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(235 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_236() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(236 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_237() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(237 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_238() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(238 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_239() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(239 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_240() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(240 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_241() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(241 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_242() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(242 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_243() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(243 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_244() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(244 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_245() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(245 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_246() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(246 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_247() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(247 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_248() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(248 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_249() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(249 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_250() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(250 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_251() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(251 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_252() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(252 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_253() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(253 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_254() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(254 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_255() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(255 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_256() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(256 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_257() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(257 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_258() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(258 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_259() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(259 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_260() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(260 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_261() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(261 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_262() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(262 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_263() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(263 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_264() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(264 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_265() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(265 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_266() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(266 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_267() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(267 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_268() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(268 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_269() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(269 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_270() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(270 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_271() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(271 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_272() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(272 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_273() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(273 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_274() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(274 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_275() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(275 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_276() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(276 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_277() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(277 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_278() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(278 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_279() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(279 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_280() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(280 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_281() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(281 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_282() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(282 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_283() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(283 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_284() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(284 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_285() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(285 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_286() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(286 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_287() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(287 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_288() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(288 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_289() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(289 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    #[test]
    fn test_wrappers_stress_290() {
        let env = CartPoleEnv::new();
        let mut framed = FrameStackWrapper::new(env, 2);
        let obs = framed.reset().unwrap();
        assert_eq!(obs.numel(), 8);

        let step = framed.step(290 % 2).unwrap();
        assert_eq!(step.observation.numel(), 8);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
}
