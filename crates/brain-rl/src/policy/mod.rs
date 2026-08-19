//! # Action Selection Policies & Exploration Schedules
//!
//! Universal Policy trait, Epsilon-Greedy policies, and linear/exponential exploration schedules.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod dist;
pub mod gaussian;

pub use dist::{CategoricalDist, DiagonalGaussianDist};
pub use gaussian::GaussianPolicy;

use brain_core::Tensor;

/// Exploration epsilon annealing schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum EpsilonSchedule {
    Constant(f64),
    Linear { start: f64, end: f64, decay_steps: usize },
    Exponential { start: f64, end: f64, decay_rate: f64 },
}

impl EpsilonSchedule {
    pub fn get_epsilon(&self, step: usize) -> f64 {
        match self {
            EpsilonSchedule::Constant(eps) => *eps,
            EpsilonSchedule::Linear { start, end, decay_steps } => {
                if step >= *decay_steps {
                    *end
                } else {
                    let progress = step as f64 / *decay_steps as f64;
                    start + (end - start) * progress
                }
            }
            EpsilonSchedule::Exponential { start, end, decay_rate } => {
                let eps = start * decay_rate.powi(step as i32);
                eps.max(*end)
            }
        }
    }
}

/// Universal policy trait for action selection.
pub trait Policy: Send + Sync {
    /// Selects an action given current observation state.
    fn act(&mut self, state: &Tensor, step: usize) -> usize;

    /// Evaluates action distribution log probability given state and action.
    fn log_prob(&self, _state: &Tensor, _action: usize) -> f64 {
        0.0
    }
}

/// Epsilon-Greedy Exploration Policy wrapping Q-value evaluations.
#[derive(Debug, Clone)]
pub struct EpsilonGreedyPolicy {
    pub schedule: EpsilonSchedule,
    pub num_actions: usize,
    pub rng_state: u64,
}

impl EpsilonGreedyPolicy {
    pub fn new(schedule: EpsilonSchedule, num_actions: usize) -> Self {
        Self {
            schedule,
            num_actions,
            rng_state: 8888,
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Selects action either greedily from Q-values or randomly via exploration.
    pub fn select_action(&mut self, q_values: &[f64], step: usize) -> usize {
        let eps = self.schedule.get_epsilon(step);
        if self.next_f64() < eps {
            (self.next_f64() * self.num_actions as f64) as usize % self.num_actions
        } else {
            let mut best_idx = 0;
            let mut best_val = f64::NEG_INFINITY;
            for (i, &v) in q_values.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            best_idx
        }
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
