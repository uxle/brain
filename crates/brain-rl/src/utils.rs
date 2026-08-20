//! # Mathematical & Statistical RL Utilities
//!
//! Return discounting, exponential moving averages, and cumulative sum computations.
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

/// Computes discounted cumulative future returns: G_t = sum_{l=0}^T gamma^l r_{t+l}.
pub fn discount_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let n = rewards.len();
    let mut returns = vec![0.0; n];
    let mut g = 0.0;
    for t in (0..n).rev() {
        g = rewards[t] + gamma * g;
        returns[t] = g;
    }
    returns
}

/// Computes exponential moving average of episodic returns.
pub fn moving_average(returns: &[f64], alpha: f64) -> Vec<f64> {
    let mut smoothed = Vec::with_capacity(returns.len());
    let mut avg = 0.0;
    for (i, &r) in returns.iter().enumerate() {
        if i == 0 {
            avg = r;
        } else {
            avg = (1.0 - alpha) * avg + alpha * r;
        }
        smoothed.push(avg);
    }
    smoothed
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
