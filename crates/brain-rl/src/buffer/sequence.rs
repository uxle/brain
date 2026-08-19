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
}
