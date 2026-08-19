//! # Label Smoothing Regularization
//!
//! Softens hard one-hot classification targets: y_k = (1 - eps) * y_k + eps / K.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;
use super::core::{RegError, RegResult};

/// Configuration for Label Smoothing.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelSmoothConfig {
    pub smoothing: f64,
    pub num_classes: usize,
}

impl Default for LabelSmoothConfig {
    fn default() -> Self {
        Self {
            smoothing: 0.1,
            num_classes: 10,
        }
    }
}

/// Label Smoothing Engine.
#[derive(Debug, Clone)]
pub struct LabelSmoothing {
    pub config: LabelSmoothConfig,
}

impl LabelSmoothing {
    pub fn new(config: LabelSmoothConfig) -> Self {
        Self { config }
    }

    /// Computes smoothed soft target distribution for one-hot integer target indices.
    pub fn smooth_targets(&self, targets: &[usize]) -> RegResult<Tensor> {
        let k = self.config.num_classes;
        if k == 0 {
            return Err(RegError::ConfigurationError("Number of classes must be > 0".into()));
        }

        let num_samples = targets.len();
        let mut out = vec![0.0; num_samples * k];
        let eps = self.config.smoothing.clamp(0.0, 1.0);
        let uniform = eps / k as f64;

        for (i, &target_idx) in targets.iter().enumerate() {
            if target_idx >= k {
                return Err(RegError::ConfigurationError(format!("Target index {} >= num_classes {}", target_idx, k)));
            }
            for c in 0..k {
                let val = if c == target_idx {
                    (1.0 - eps) + uniform
                } else {
                    uniform
                };
                out[i * k + c] = val;
            }
        }

        Ok(Tensor::from_slice(&out, vec![num_samples, k]))
    }
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
