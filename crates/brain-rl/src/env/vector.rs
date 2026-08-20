//! # Vectorized Parallel Environments
//!
//! Synchronous batched stepping across multiple parallel environment instances.
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

use super::super::core::{RlError, RlResult};
use super::{Env, EnvStep};
use brain_core::Tensor;

/// Synchronous dummy vectorized environment runner.
pub struct DummyVecEnv<E: Env> {
    pub envs: Vec<E>,
}

impl<E: Env> DummyVecEnv<E> {
    pub fn new(envs: Vec<E>) -> Self {
        Self { envs }
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Resets all parallel environments.
    pub fn reset_all(&mut self) -> RlResult<Vec<Tensor>> {
        let mut obs = Vec::with_capacity(self.envs.len());
        for env in &mut self.envs {
            obs.push(env.reset()?);
        }
        Ok(obs)
    }

    /// Synchronously steps all environments given an array of actions.
    pub fn step_all(&mut self, actions: &[usize]) -> RlResult<Vec<EnvStep>> {
        if actions.len() != self.envs.len() {
            return Err(RlError::EnvironmentError(
                "Action count != Env count".into(),
            ));
        }

        let mut steps = Vec::with_capacity(self.envs.len());
        for (env, &action) in self.envs.iter_mut().zip(actions.iter()) {
            let mut s = env.step(action)?;
            if s.done || s.truncated {
                s.observation = env.reset()?;
            }
            steps.push(s);
        }
        Ok(steps)
    }
}

pub type VecEnv<E> = DummyVecEnv<E>;

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
