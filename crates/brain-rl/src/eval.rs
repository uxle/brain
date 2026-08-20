//! # Deterministic Agent Evaluation & Metrics
//!
//! Evaluates policy return distributions, standard deviations, and success rates across test episodes.
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

use super::core::{RlError, RlResult};
use super::dqn::DqnAgent;
use super::env::Env;

/// Comprehensive evaluation report.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    pub num_episodes: usize,
    pub mean_return: f64,
    pub std_return: f64,
    pub min_return: f64,
    pub max_return: f64,
}

/// Evaluates DQN agent deterministically over multiple test episodes.
pub fn evaluate_dqn<E: Env>(
    agent: &mut DqnAgent,
    env: &mut E,
    num_episodes: usize,
) -> RlResult<EvalReport> {
    if num_episodes == 0 {
        return Err(RlError::EnvironmentError("num_episodes must be > 0".into()));
    }

    let mut returns = Vec::with_capacity(num_episodes);

    for _ in 0..num_episodes {
        let mut state = env.reset()?;
        let mut ep_return = 0.0;

        for _ in 0..1000 {
            let q_values = agent.q_online.forward(&state);
            let mut best_a = 0;
            let mut best_q = f64::NEG_INFINITY;
            for (a, &v) in q_values.iter().enumerate() {
                if v > best_q {
                    best_q = v;
                    best_a = a;
                }
            }

            let step = env.step(best_a)?;
            ep_return += step.reward;
            state = step.observation;

            if step.done || step.truncated {
                break;
            }
        }

        returns.push(ep_return);
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let min = returns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(EvalReport {
        num_episodes,
        mean_return: mean,
        std_return: std,
        min_return: min,
        max_return: max,
    })
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
