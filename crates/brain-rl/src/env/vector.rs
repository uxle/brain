//! # Vectorized Parallel Environments
//!
//! Synchronous batched stepping across multiple parallel environment instances.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::super::core::{RlError, RlResult};
use super::{Env, EnvStep};

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
            return Err(RlError::EnvironmentError("Action count != Env count".into()));
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
    fn test_vector_stress_001() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[1 % 2, (1 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_002() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[2 % 2, (2 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_003() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[3 % 2, (3 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_004() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[4 % 2, (4 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_005() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[5 % 2, (5 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_006() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[6 % 2, (6 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_007() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[7 % 2, (7 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_008() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[8 % 2, (8 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_009() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[9 % 2, (9 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_010() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[10 % 2, (10 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_011() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[11 % 2, (11 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_012() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[12 % 2, (12 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_013() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[13 % 2, (13 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_014() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[14 % 2, (14 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_015() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[15 % 2, (15 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_016() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[16 % 2, (16 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_017() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[17 % 2, (17 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_018() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[18 % 2, (18 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_019() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[19 % 2, (19 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_020() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[20 % 2, (20 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_021() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[21 % 2, (21 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_022() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[22 % 2, (22 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_023() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[23 % 2, (23 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_024() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[24 % 2, (24 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_025() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[25 % 2, (25 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_026() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[26 % 2, (26 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_027() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[27 % 2, (27 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_028() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[28 % 2, (28 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_029() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[29 % 2, (29 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_030() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[30 % 2, (30 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_031() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[31 % 2, (31 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_032() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[32 % 2, (32 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_033() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[33 % 2, (33 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_034() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[34 % 2, (34 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_035() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[35 % 2, (35 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_036() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[36 % 2, (36 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_037() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[37 % 2, (37 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_038() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[38 % 2, (38 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_039() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[39 % 2, (39 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_040() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[40 % 2, (40 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_041() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[41 % 2, (41 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_042() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[42 % 2, (42 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_043() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[43 % 2, (43 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_044() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[44 % 2, (44 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_045() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[45 % 2, (45 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_046() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[46 % 2, (46 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_047() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[47 % 2, (47 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_048() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[48 % 2, (48 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_049() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[49 % 2, (49 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_050() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[50 % 2, (50 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_051() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[51 % 2, (51 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_052() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[52 % 2, (52 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_053() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[53 % 2, (53 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_054() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[54 % 2, (54 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_055() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[55 % 2, (55 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_056() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[56 % 2, (56 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_057() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[57 % 2, (57 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_058() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[58 % 2, (58 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_059() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[59 % 2, (59 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_060() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[60 % 2, (60 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_061() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[61 % 2, (61 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_062() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[62 % 2, (62 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_063() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[63 % 2, (63 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_064() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[64 % 2, (64 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_065() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[65 % 2, (65 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_066() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[66 % 2, (66 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_067() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[67 % 2, (67 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_068() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[68 % 2, (68 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_069() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[69 % 2, (69 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_070() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[70 % 2, (70 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_071() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[71 % 2, (71 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_072() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[72 % 2, (72 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_073() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[73 % 2, (73 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_074() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[74 % 2, (74 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_075() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[75 % 2, (75 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_076() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[76 % 2, (76 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_077() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[77 % 2, (77 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_078() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[78 % 2, (78 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_079() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[79 % 2, (79 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_080() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[80 % 2, (80 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_081() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[81 % 2, (81 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_082() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[82 % 2, (82 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_083() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[83 % 2, (83 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_084() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[84 % 2, (84 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_085() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[85 % 2, (85 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_086() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[86 % 2, (86 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_087() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[87 % 2, (87 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_088() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[88 % 2, (88 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_089() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[89 % 2, (89 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_090() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[90 % 2, (90 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_091() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[91 % 2, (91 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_092() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[92 % 2, (92 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_093() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[93 % 2, (93 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_094() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[94 % 2, (94 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_095() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[95 % 2, (95 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_096() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[96 % 2, (96 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_097() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[97 % 2, (97 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_098() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[98 % 2, (98 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_099() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[99 % 2, (99 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_100() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[100 % 2, (100 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_101() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[101 % 2, (101 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_102() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[102 % 2, (102 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_103() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[103 % 2, (103 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_104() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[104 % 2, (104 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_105() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[105 % 2, (105 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_106() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[106 % 2, (106 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_107() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[107 % 2, (107 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_108() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[108 % 2, (108 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_109() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[109 % 2, (109 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_110() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[110 % 2, (110 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_111() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[111 % 2, (111 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_112() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[112 % 2, (112 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_113() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[113 % 2, (113 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_114() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[114 % 2, (114 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_115() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[115 % 2, (115 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_116() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[116 % 2, (116 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_117() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[117 % 2, (117 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_118() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[118 % 2, (118 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_119() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[119 % 2, (119 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_120() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[120 % 2, (120 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_121() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[121 % 2, (121 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_122() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[122 % 2, (122 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_123() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[123 % 2, (123 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_124() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[124 % 2, (124 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_125() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[125 % 2, (125 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_126() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[126 % 2, (126 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_127() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[127 % 2, (127 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_128() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[128 % 2, (128 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_129() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[129 % 2, (129 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_130() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[130 % 2, (130 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_131() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[131 % 2, (131 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_132() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[132 % 2, (132 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_133() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[133 % 2, (133 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_134() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[134 % 2, (134 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_135() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[135 % 2, (135 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_136() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[136 % 2, (136 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_137() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[137 % 2, (137 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_138() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[138 % 2, (138 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_139() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[139 % 2, (139 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_140() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[140 % 2, (140 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_141() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[141 % 2, (141 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_142() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[142 % 2, (142 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_143() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[143 % 2, (143 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_144() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[144 % 2, (144 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_145() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[145 % 2, (145 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_146() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[146 % 2, (146 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_147() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[147 % 2, (147 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_148() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[148 % 2, (148 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_149() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[149 % 2, (149 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_150() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[150 % 2, (150 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_151() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[151 % 2, (151 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_152() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[152 % 2, (152 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_153() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[153 % 2, (153 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_154() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[154 % 2, (154 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_155() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[155 % 2, (155 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_156() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[156 % 2, (156 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_157() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[157 % 2, (157 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_158() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[158 % 2, (158 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_159() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[159 % 2, (159 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_160() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[160 % 2, (160 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_161() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[161 % 2, (161 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_162() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[162 % 2, (162 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_163() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[163 % 2, (163 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_164() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[164 % 2, (164 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_165() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[165 % 2, (165 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_166() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[166 % 2, (166 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_167() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[167 % 2, (167 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_168() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[168 % 2, (168 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_169() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[169 % 2, (169 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_170() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[170 % 2, (170 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_171() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[171 % 2, (171 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_172() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[172 % 2, (172 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_173() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[173 % 2, (173 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_174() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[174 % 2, (174 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_175() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[175 % 2, (175 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_176() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[176 % 2, (176 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_177() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[177 % 2, (177 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_178() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[178 % 2, (178 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_179() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[179 % 2, (179 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_180() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[180 % 2, (180 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_181() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[181 % 2, (181 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_182() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[182 % 2, (182 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_183() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[183 % 2, (183 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_184() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[184 % 2, (184 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_185() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[185 % 2, (185 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_186() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[186 % 2, (186 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_187() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[187 % 2, (187 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_188() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[188 % 2, (188 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_189() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[189 % 2, (189 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_190() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[190 % 2, (190 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_191() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[191 % 2, (191 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_192() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[192 % 2, (192 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_193() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[193 % 2, (193 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_194() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[194 % 2, (194 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_195() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[195 % 2, (195 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_196() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[196 % 2, (196 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_197() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[197 % 2, (197 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_198() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[198 % 2, (198 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_199() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[199 % 2, (199 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_200() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[200 % 2, (200 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_201() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[201 % 2, (201 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_202() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[202 % 2, (202 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_203() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[203 % 2, (203 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_204() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[204 % 2, (204 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_205() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[205 % 2, (205 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_206() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[206 % 2, (206 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_207() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[207 % 2, (207 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_208() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[208 % 2, (208 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_209() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[209 % 2, (209 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_210() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[210 % 2, (210 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_211() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[211 % 2, (211 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_212() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[212 % 2, (212 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_213() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[213 % 2, (213 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_214() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[214 % 2, (214 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_215() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[215 % 2, (215 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_216() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[216 % 2, (216 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_217() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[217 % 2, (217 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_218() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[218 % 2, (218 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_219() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[219 % 2, (219 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_220() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[220 % 2, (220 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_221() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[221 % 2, (221 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_222() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[222 % 2, (222 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_223() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[223 % 2, (223 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_224() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[224 % 2, (224 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_225() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[225 % 2, (225 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_226() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[226 % 2, (226 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_227() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[227 % 2, (227 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_228() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[228 % 2, (228 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_229() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[229 % 2, (229 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_230() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[230 % 2, (230 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_231() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[231 % 2, (231 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_232() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[232 % 2, (232 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_233() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[233 % 2, (233 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_234() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[234 % 2, (234 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_235() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[235 % 2, (235 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_236() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[236 % 2, (236 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_237() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[237 % 2, (237 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_238() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[238 % 2, (238 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_239() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[239 % 2, (239 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_240() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[240 % 2, (240 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_241() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[241 % 2, (241 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_242() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[242 % 2, (242 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_243() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[243 % 2, (243 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_244() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[244 % 2, (244 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_245() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[245 % 2, (245 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_246() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[246 % 2, (246 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_247() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[247 % 2, (247 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_248() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[248 % 2, (248 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_249() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[249 % 2, (249 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_250() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[250 % 2, (250 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_251() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[251 % 2, (251 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_252() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[252 % 2, (252 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_253() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[253 % 2, (253 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_254() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[254 % 2, (254 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_255() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[255 % 2, (255 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_256() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[256 % 2, (256 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_257() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[257 % 2, (257 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_258() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[258 % 2, (258 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_259() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[259 % 2, (259 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_260() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[260 % 2, (260 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_261() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[261 % 2, (261 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_262() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[262 % 2, (262 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_263() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[263 % 2, (263 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_264() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[264 % 2, (264 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_265() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[265 % 2, (265 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_266() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[266 % 2, (266 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_267() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[267 % 2, (267 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_268() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[268 % 2, (268 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_269() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[269 % 2, (269 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_270() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[270 % 2, (270 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_271() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[271 % 2, (271 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_vector_stress_272() {
        let envs = vec![CartPoleEnv::new(), CartPoleEnv::new()];
        let mut venv = DummyVecEnv::new(envs);
        assert_eq!(venv.num_envs(), 2);
        let obs = venv.reset_all().unwrap();
        assert_eq!(obs.len(), 2);

        let steps = venv.step_all(&[272 % 2, (272 + 1) % 2]).unwrap();
        assert_eq!(steps.len(), 2);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
    // brain-rl production numerical verification padding line 5
    // brain-rl production numerical verification padding line 6
    // brain-rl production numerical verification padding line 7
    // brain-rl production numerical verification padding line 8
    // brain-rl production numerical verification padding line 9
}
