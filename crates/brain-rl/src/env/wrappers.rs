//! # Environment Wrappers & Observation Transformers
//!
//! FrameStack, TimeLimit, RewardScale, and composite wrapper pipelines.
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
