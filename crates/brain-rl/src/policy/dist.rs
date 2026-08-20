//! # Policy Probability Distributions
//!
//! Discrete Categorical and Continuous Diagonal Gaussian action distributions with log-probability and entropy.
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

/// Discrete Categorical Distribution parameterized by unnormalized logits.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalDist {
    pub logits: Vec<f64>,
    pub probs: Vec<f64>,
}

impl CategoricalDist {
    pub fn from_logits(logits: &[f64]) -> Self {
        let mut max_l = f64::NEG_INFINITY;
        for &l in logits {
            if l > max_l {
                max_l = l;
            }
        }

        let mut sum_exp = 0.0;
        let mut exp_logits = Vec::with_capacity(logits.len());
        for &l in logits {
            let e = (l - max_l).exp();
            exp_logits.push(e);
            sum_exp += e;
        }

        let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum_exp).collect();
        Self {
            logits: logits.to_vec(),
            probs,
        }
    }

    /// Computes log-probability of discrete action.
    pub fn log_prob(&self, action: usize) -> f64 {
        if action < self.probs.len() {
            self.probs[action].max(1e-15).ln()
        } else {
            -1e10
        }
    }

    /// Computes Shannon entropy.
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &p in &self.probs {
            if p > 1e-15 {
                h -= p * p.ln();
            }
        }
        h
    }
}

/// Diagonal Gaussian Distribution for continuous action vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalGaussianDist {
    pub mean: Vec<f64>,
    pub log_std: Vec<f64>,
}

impl DiagonalGaussianDist {
    pub fn new(mean: Vec<f64>, log_std: Vec<f64>) -> Self {
        Self { mean, log_std }
    }

    /// Evaluates log probability density.
    pub fn log_prob(&self, action: &[f64]) -> f64 {
        let mut total = 0.0;
        for i in 0..self.mean.len() {
            let std = self.log_std[i].exp();
            let var = std * std;
            let diff = action[i] - self.mean[i];
            let term = -0.5 * (diff * diff / var + (2.0 * std::f64::consts::PI * var).ln());
            total += term;
        }
        total
    }

    /// Computes differential entropy of diagonal Gaussian.
    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &ls in &self.log_std {
            h += 0.5 + 0.5 * (2.0 * std::f64::consts::PI).ln() + ls;
        }
        h
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
