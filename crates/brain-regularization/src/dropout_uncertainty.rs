//! # Monte Carlo (MC) Dropout Uncertainty Estimation
//!
//! Performs stochastic forward sampling at test-time to estimate predictive mean and epistemic variance.
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
use brain_core::Tensor;

/// Configuration for MC-Dropout uncertainty estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct McDropoutConfig {
    pub num_samples: usize,
    pub confidence_level: f64,
}

impl Default for McDropoutConfig {
    fn default() -> Self {
        Self {
            num_samples: 30,
            confidence_level: 0.95,
        }
    }
}

/// Uncertainty estimation output metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct McDropoutResult {
    pub mean: Tensor,
    pub variance: Tensor,
    pub std_dev: Tensor,
}

/// Aggregates multiple stochastic model evaluation samples to compute mean and epistemic variance.
pub fn compute_mc_dropout_statistics(samples: &[Tensor]) -> RegResult<McDropoutResult> {
    if samples.is_empty() {
        return Err(RegError::EmptyTensor);
    }

    let num_samples = samples.len();
    let shape = samples[0].shape();
    let numel = samples[0].numel();

    for s in samples {
        if s.shape() != shape {
            return Err(RegError::ShapeMismatch {
                expected: shape.to_vec(),
                found: s.shape().to_vec(),
            });
        }
    }

    let mut sum_data = vec![0.0; numel];
    let mut sum_sq_data = vec![0.0; numel];

    for s in samples {
        let d = s.data();
        for i in 0..numel {
            let v = d[i];
            sum_data[i] += v;
            sum_sq_data[i] += v * v;
        }
    }

    let mut mean_data = vec![0.0; numel];
    let mut var_data = vec![0.0; numel];
    let mut std_data = vec![0.0; numel];

    let n = num_samples as f64;
    for i in 0..numel {
        let m = sum_data[i] / n;
        mean_data[i] = m;
        let v = (sum_sq_data[i] / n - m * m).max(0.0);
        var_data[i] = v;
        std_data[i] = v.sqrt();
    }

    Ok(McDropoutResult {
        mean: Tensor::from_slice(&mean_data, shape.to_vec()),
        variance: Tensor::from_slice(&var_data, shape.to_vec()),
        std_dev: Tensor::from_slice(&std_data, shape.to_vec()),
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
