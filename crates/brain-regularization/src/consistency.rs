//! # Consistency Regularization
//!
//! Enforces model output invariance under stochastic input perturbations (Pi-model / Mean Teacher style).
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::core::{RegError, RegResult};

/// Configuration for consistency regularization.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsistencyConfig {
    pub weight: f64,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self { weight: 1.0 }
    }
}

/// Evaluates Mean Squared Error consistency penalty between two stochastic predictions.
pub fn compute_consistency_loss(pred1: &Tensor, pred2: &Tensor, weight: f64) -> RegResult<f64> {
    if pred1.shape() != pred2.shape() {
        return Err(RegError::ShapeMismatch {
            expected: pred1.shape().to_vec(),
            found: pred2.shape().to_vec(),
        });
    }

    let d1 = pred1.data();
    let d2 = pred2.data();
    let mut sum_sq = 0.0;

    for i in 0..d1.len() {
        let diff = d1[i] - d2[i];
        sum_sq += diff * diff;
    }

    let mse = sum_sq / d1.len().max(1) as f64;
    Ok(weight * mse)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::dropout::*;
    use crate::normalization::*;
    use crate::regularizers::*;
    use crate::decay::*;
    use crate::earlystop::*;
    use crate::stopping::*;
    use crate::augment::*;
    use crate::perturb::*;
    use crate::dropout_uncertainty::*;
    use crate::label_smooth::*;
    use crate::curriculum::*;
    use crate::consistency::*;
    use crate::rules::*;
    use crate::registry::*;
    use crate::train_hooks::*;
    use crate::ops::*;
    use crate::r#impl::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
