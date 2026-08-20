//! # Concrete & Adaptive Dropout
//!
//! Continuous relaxation of dropout with temperature annealing and learned parameter p (Gal & Ghahramani).
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

use super::super::core::{RegKind, RegResult, Regularization};
use super::super::utils::XorShift64;
use brain_core::Tensor;

/// Configuration for Concrete Dropout.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveDropoutConfig {
    pub temperature: f64,
    pub weight_regularizer: f64,
    pub dropout_regularizer: f64,
}

impl Default for AdaptiveDropoutConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            weight_regularizer: 1e-6,
            dropout_regularizer: 1e-5,
        }
    }
}

/// Concrete Dropout Layer.
#[derive(Debug, Clone)]
pub struct ConcreteDropout {
    pub logit_p: f64,
    pub temperature: f64,
    pub is_training: bool,
    pub rng: XorShift64,
}

impl ConcreteDropout {
    pub fn new(initial_p: f64, temperature: f64) -> Self {
        let p = initial_p.clamp(1e-4, 1.0 - 1e-4);
        let logit_p = (p / (1.0 - p)).ln();
        Self {
            logit_p,
            temperature: temperature.max(1e-4),
            is_training: true,
            rng: XorShift64::new(999),
        }
    }

    /// Computes current retention probability.
    pub fn current_p(&self) -> f64 {
        1.0 / (1.0 + (-self.logit_p).exp())
    }
}

impl Regularization for ConcreteDropout {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        if !self.is_training {
            return Ok(input.clone());
        }

        let p = self.current_p();
        let data = input.data();
        let n = data.len();
        let mut out_data = vec![0.0; n];

        let temp = self.temperature;
        let eps = 1e-7;

        for i in 0..n {
            let u = self.rng.next_f64().clamp(eps, 1.0 - eps);
            let drop_prob = ((self.logit_p + (u / (1.0 - u)).ln()) / temp).clamp(-20.0, 20.0);
            let z = 1.0 / (1.0 + (-drop_prob).exp());
            out_data[i] = data[i] * z / (1.0 - p);
        }

        Ok(Tensor::from_slice(&out_data, input.shape().to_vec()))
    }

    fn train_mode(&mut self) {
        self.is_training = true;
    }

    fn eval_mode(&mut self) {
        self.is_training = false;
    }

    fn kind(&self) -> RegKind {
        RegKind::Dropout
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
