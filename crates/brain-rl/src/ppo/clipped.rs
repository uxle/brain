//! # PPO Clipped Surrogate Loss Objective
//!
//! Evaluates clipped policy surrogate loss and clipped value function error.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// PPO Clipped Surrogate Objective Evaluator.
#[derive(Debug, Clone)]
pub struct PpoClippedObjective {
    pub clip_epsilon: f64,
}

impl PpoClippedObjective {
    pub fn new(clip_epsilon: f64) -> Self {
        Self {
            clip_epsilon: clip_epsilon.max(0.0),
        }
    }

    /// Computes clipped policy loss term given probability ratio r_t and estimated advantage A_t.
    pub fn compute_policy_loss(&self, ratio: f64, advantage: f64) -> f64 {
        let surr1 = ratio * advantage;
        let surr2 = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage;
        -surr1.min(surr2)
    }

    /// Computes clipped value function loss.
    pub fn compute_value_loss(&self, v_pred: f64, v_old: f64, v_target: f64) -> f64 {
        let v_clipped = v_old + (v_pred - v_old).clamp(-self.clip_epsilon, self.clip_epsilon);
        let loss1 = (v_pred - v_target).powi(2);
        let loss2 = (v_clipped - v_target).powi(2);
        0.5 * loss1.max(loss2)
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
