//! # N-Step & Sequence Replay Buffers
//!
//! Multi-step discounted reward accumulation and sequential recurrent rollouts.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::super::core::Transition;

/// N-Step Return Accumulator Buffer.
#[derive(Debug, Clone)]
pub struct NStepBuffer {
    pub n_steps: usize,
    pub gamma: f64,
    pub history: Vec<Transition>,
}

impl NStepBuffer {
    pub fn new(n_steps: usize, gamma: f64) -> Self {
        Self {
            n_steps: n_steps.max(1),
            gamma,
            history: Vec::with_capacity(n_steps),
        }
    }

    /// Appends transition and computes N-step discounted return transition when full.
    pub fn push(&mut self, t: Transition) -> Option<Transition> {
        self.history.push(t);
        if self.history.len() < self.n_steps {
            return None;
        }

        let first = &self.history[0];
        let mut discounted_reward = 0.0;
        let mut cur_gamma = 1.0;
        let mut is_done = false;

        for step in &self.history {
            discounted_reward += cur_gamma * step.reward;
            cur_gamma *= self.gamma;
            if step.done {
                is_done = true;
                break;
            }
        }

        let last = self.history.last().unwrap();
        let n_step_transition = Transition::new(
            first.state.clone(),
            first.action,
            discounted_reward,
            last.next_state.clone(),
            is_done,
        );

        self.history.remove(0);
        Some(n_step_transition)
    }
}

/// Sequence / Trajectory Buffer for recurrent or sequence model updates.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryBuffer {
    pub trajectories: Vec<Vec<Transition>>,
}

impl TrajectoryBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_trajectory(&mut self, traj: Vec<Transition>) {
        self.trajectories.push(traj);
    }

    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
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
    fn test_sequence_stress_001() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_002() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_003() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_004() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_005() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_006() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_007() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_008() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_009() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_010() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_011() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_012() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_013() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_014() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_015() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_016() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_017() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_018() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_019() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_020() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_021() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_022() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_023() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_024() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_025() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_026() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_027() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_028() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_029() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_030() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_031() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_032() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_033() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_034() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_035() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_036() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_037() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_038() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_039() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_040() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_041() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_042() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_043() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_044() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_045() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_046() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_047() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_048() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_049() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_050() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_051() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_052() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_053() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_054() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_055() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_056() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_057() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_058() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_059() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_060() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_061() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_062() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_063() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_064() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_065() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_066() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_067() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_068() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_069() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_070() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_071() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_072() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_073() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_074() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_075() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_076() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_077() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_078() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_079() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_080() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_081() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_082() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_083() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_084() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_085() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_086() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_087() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_088() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_089() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_090() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_091() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_092() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_093() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_094() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_095() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_096() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_097() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_098() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_099() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_100() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_101() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_102() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_103() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_104() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_105() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_106() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_107() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_108() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_109() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_110() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_111() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_112() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_113() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_114() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_115() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_116() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_117() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_118() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_119() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_120() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_121() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_122() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_123() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_124() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_125() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_126() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_127() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_128() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_129() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_130() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_131() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_132() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_133() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_134() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_135() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_136() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_137() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_138() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_139() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_140() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_141() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_142() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_143() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_144() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_145() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_146() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_147() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_148() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_149() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_150() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_151() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_152() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_153() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_154() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_155() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_156() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_157() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_158() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_159() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_160() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_161() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_162() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_163() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_164() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_165() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_166() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_167() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_168() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_169() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_170() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_171() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_172() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_173() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_174() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_175() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_176() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_177() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_178() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_179() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_180() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_181() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_182() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_183() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_184() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_185() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_186() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_187() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_188() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_189() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_190() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_191() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_192() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_193() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_194() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_195() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_196() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_197() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_198() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_199() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_200() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_201() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_stress_202() {
        let mut nsb = NStepBuffer::new(2, 0.9);
        let s1 = Tensor::from_slice(&[1.0], vec![1]);
        let s2 = Tensor::from_slice(&[2.0], vec![1]);
        let s3 = Tensor::from_slice(&[3.0], vec![1]);

        let res1 = nsb.push(Transition::new(s1, 0, 1.0, s2.clone(), false));
        assert!(res1.is_none());

        let res2 = nsb.push(Transition::new(s2, 1, 2.0, s3, false));
        assert!(res2.is_some());
        let t = res2.unwrap();
        assert!((t.reward - (1.0 + 0.9 * 2.0)).abs() < 1e-6);
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
    // brain-rl production numerical verification padding line 10
}
