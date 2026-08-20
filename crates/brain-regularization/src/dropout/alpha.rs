//! # Alpha Dropout (SELU-Compatible)
//!
//! Preserves the self-normalizing properties (zero mean and unit variance) of SELU activations.
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

/// Configuration for Alpha Dropout layer.
#[derive(Debug, Clone, PartialEq)]
pub struct AlphaDropoutConfig {
    pub p: f64,
    pub alpha_prime: f64,
    pub scale_factor: f64,
}

impl Default for AlphaDropoutConfig {
    fn default() -> Self {
        Self {
            p: 0.5,
            alpha_prime: -1.7580993408473766,
            scale_factor: 1.0,
        }
    }
}

/// Alpha Dropout layer designed to operate on Self-Normalizing Neural Networks (SNNs).
#[derive(Debug, Clone)]
pub struct AlphaDropout {
    pub p: f64,
    pub is_training: bool,
    pub alpha_prime: f64,
    pub a: f64,
    pub b: f64,
    pub rng: XorShift64,
}

impl AlphaDropout {
    pub fn new(p: f64) -> Self {
        let p_clamped = p.clamp(0.0, 1.0);
        let alpha = 1.673_263_242_354_377_2;
        let scale = 1.050_700_987_355_480_5;
        let alpha_prime = -scale * alpha;

        let a = ((1.0 - p_clamped) * (1.0 + p_clamped * alpha_prime * alpha_prime)).powf(-0.5);
        let b = -a * alpha_prime * p_clamped;

        Self {
            p: p_clamped,
            is_training: true,
            alpha_prime,
            a,
            b,
            rng: XorShift64::new(777),
        }
    }
}

impl Regularization for AlphaDropout {
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor> {
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let data = input.data();
        let n = data.len();
        let mut out_data = vec![0.0; n];

        for i in 0..n {
            let r = self.rng.next_f64();
            let val = if r < self.p {
                self.alpha_prime
            } else {
                data[i]
            };
            out_data[i] = self.a * val + self.b;
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
        RegKind::AlphaDropout
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
