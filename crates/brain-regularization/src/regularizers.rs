//! # Explicit Penalty Regularizers
//!
//! L1 (Lasso), L2 (Ridge), Elastic Net (L1 + L2), and Huber robust penalty regularizers.
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

use brain_core::Tensor;

/// Configuration for penalty regularizers.
#[derive(Debug, Clone, PartialEq)]
pub struct RegularizerConfig {
    pub l1_factor: f64,
    pub l2_factor: f64,
    pub huber_delta: f64,
}

impl Default for RegularizerConfig {
    fn default() -> Self {
        Self {
            l1_factor: 1e-4,
            l2_factor: 1e-4,
            huber_delta: 1.0,
        }
    }
}

/// Fundamental trait for explicit parameter penalty regularizers.
pub trait Regularizer: Send + Sync {
    /// Computes penalty scalar loss contribution for model parameters.
    fn penalty(&self, params: &[Tensor]) -> f64;

    /// Computes regularization gradient penalty term added to parameter gradients.
    fn grad_penalty(&self, param: &Tensor) -> Tensor;
}

/// L1 (Lasso) Regularizer enforcing sparsity in parameter tensors.
#[derive(Debug, Clone)]
pub struct L1Regularizer {
    pub factor: f64,
}

impl L1Regularizer {
    pub fn new(factor: f64) -> Self {
        Self {
            factor: factor.max(0.0),
        }
    }
}

impl Regularizer for L1Regularizer {
    fn penalty(&self, params: &[Tensor]) -> f64 {
        let mut total = 0.0;
        for p in params {
            for &v in p.data() {
                total += v.abs();
            }
        }
        self.factor * total
    }

    fn grad_penalty(&self, param: &Tensor) -> Tensor {
        let data = param.data();
        let mut g = vec![0.0; data.len()];
        for i in 0..data.len() {
            let v = data[i];
            let sign = if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            };
            g[i] = self.factor * sign;
        }
        Tensor::from_slice(&g, param.shape().to_vec())
    }
}

/// L2 (Ridge) Regularizer penalizing large weight magnitudes.
#[derive(Debug, Clone)]
pub struct L2Regularizer {
    pub factor: f64,
}

impl L2Regularizer {
    pub fn new(factor: f64) -> Self {
        Self {
            factor: factor.max(0.0),
        }
    }
}

impl Regularizer for L2Regularizer {
    fn penalty(&self, params: &[Tensor]) -> f64 {
        let mut total = 0.0;
        for p in params {
            for &v in p.data() {
                total += v * v;
            }
        }
        0.5 * self.factor * total
    }

    fn grad_penalty(&self, param: &Tensor) -> Tensor {
        let data = param.data();
        let mut g = vec![0.0; data.len()];
        for i in 0..data.len() {
            g[i] = self.factor * data[i];
        }
        Tensor::from_slice(&g, param.shape().to_vec())
    }
}

/// Elastic Net Regularizer combining L1 and L2 penalties.
#[derive(Debug, Clone)]
pub struct ElasticNetRegularizer {
    pub l1: L1Regularizer,
    pub l2: L2Regularizer,
}

impl ElasticNetRegularizer {
    pub fn new(l1_factor: f64, l2_factor: f64) -> Self {
        Self {
            l1: L1Regularizer::new(l1_factor),
            l2: L2Regularizer::new(l2_factor),
        }
    }
}

impl Regularizer for ElasticNetRegularizer {
    fn penalty(&self, params: &[Tensor]) -> f64 {
        self.l1.penalty(params) + self.l2.penalty(params)
    }

    fn grad_penalty(&self, param: &Tensor) -> Tensor {
        let g1 = self.l1.grad_penalty(param);
        let g2 = self.l2.grad_penalty(param);
        let d1 = g1.data();
        let d2 = g2.data();
        let mut g = vec![0.0; d1.len()];
        for i in 0..d1.len() {
            g[i] = d1[i] + d2[i];
        }
        Tensor::from_slice(&g, param.shape().to_vec())
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
        clippy::doc_markdown
    )]
    use super::*;
    use crate::augment::*;
    use crate::config::*;
    use crate::consistency::*;
    use crate::core::*;
    use crate::curriculum::*;
    use crate::decay::*;
    use crate::dropout::*;
    use crate::dropout_uncertainty::*;
    use crate::earlystop::*;
    use crate::label_smooth::*;
    use crate::normalization::*;
    use crate::ops::*;
    use crate::perturb::*;
    use crate::r#impl::*;
    use crate::registry::*;
    use crate::regularizers::*;
    use crate::rules::*;
    use crate::stopping::*;
    use crate::train_hooks::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
