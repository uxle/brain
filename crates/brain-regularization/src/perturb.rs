//! # Noise Injection & Adversarial Perturbation
//!
//! Gaussian/Uniform jitter injection and Fast Gradient Sign Method (FGSM) adversarial regularization.
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

use super::core::{RegError, RegResult};
use super::utils::XorShift64;
use brain_core::Tensor;

/// Configuration for noise and perturbation transforms.
#[derive(Debug, Clone, PartialEq)]
pub struct PerturbConfig {
    pub noise_std: f64,
    pub fgsm_epsilon: f64,
}

impl Default for PerturbConfig {
    fn default() -> Self {
        Self {
            noise_std: 0.01,
            fgsm_epsilon: 0.05,
        }
    }
}

/// Injects Gaussian noise directly into activation or parameter tensors during training.
#[derive(Debug, Clone)]
pub struct GaussianNoise {
    pub std_dev: f64,
    pub rng: XorShift64,
}

impl GaussianNoise {
    pub fn new(std_dev: f64) -> Self {
        Self {
            std_dev: std_dev.max(0.0),
            rng: XorShift64::new(303),
        }
    }

    pub fn inject(&mut self, tensor: &Tensor) -> Tensor {
        if self.std_dev == 0.0 {
            return tensor.clone();
        }

        let data = tensor.data();
        let mut out = vec![0.0; data.len()];

        for i in 0..data.len() {
            let noise = self.rng.next_gaussian() * self.std_dev;
            out[i] = data[i] + noise;
        }

        Tensor::from_slice(&out, tensor.shape().to_vec())
    }
}

/// Fast Gradient Sign Method (FGSM) adversarial input perturbation.
pub fn apply_fgsm_perturbation(input: &Tensor, grad: &Tensor, epsilon: f64) -> RegResult<Tensor> {
    if input.shape() != grad.shape() {
        return Err(RegError::ShapeMismatch {
            expected: input.shape().to_vec(),
            found: grad.shape().to_vec(),
        });
    }

    let in_data = input.data();
    let g_data = grad.data();
    let mut out = vec![0.0; in_data.len()];

    for i in 0..in_data.len() {
        let sign = if g_data[i] > 0.0 {
            1.0
        } else if g_data[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        out[i] = in_data[i] + epsilon * sign;
    }

    Ok(Tensor::from_slice(&out, input.shape().to_vec()))
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
