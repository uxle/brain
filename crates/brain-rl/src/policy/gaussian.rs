//! # Continuous Gaussian Policy
//!
//! State-conditioned Gaussian policy for continuous control action outputs.
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

use super::dist::DiagonalGaussianDist;

/// Continuous Gaussian Policy.
#[derive(Debug, Clone)]
pub struct GaussianPolicy {
    pub action_dim: usize,
    pub log_std: Vec<f64>,
}

impl GaussianPolicy {
    pub fn new(action_dim: usize) -> Self {
        Self {
            action_dim,
            log_std: vec![0.0; action_dim],
        }
    }

    /// Evaluates action distribution given state-dependent mean action output.
    pub fn distribution(&self, mean: &[f64]) -> DiagonalGaussianDist {
        DiagonalGaussianDist::new(mean.to_vec(), self.log_std.clone())
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
