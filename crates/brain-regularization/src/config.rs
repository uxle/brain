//! # Regularization Configuration Architecture
//!
//! Comprehensive hyperparameter specifications for dropout, normalization, penalties, and early stopping.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use super::core::{RegError, RegResult};

/// Universal aggregated regularization configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RegConfig {
    pub dropout_p: f64,
    pub weight_decay: f64,
    pub l1_ratio: f64,
    pub bn_momentum: f64,
    pub bn_eps: f64,
    pub early_stopping_patience: usize,
    pub early_stopping_min_delta: f64,
}

impl Default for RegConfig {
    fn default() -> Self {
        Self {
            dropout_p: 0.5,
            weight_decay: 1e-4,
            l1_ratio: 0.0,
            bn_momentum: 0.1,
            bn_eps: 1e-5,
            early_stopping_patience: 10,
            early_stopping_min_delta: 1e-4,
        }
    }
}

impl RegConfig {
    /// Validates all hyperparameter ranges.
    pub fn validate(&self) -> RegResult<()> {
        if self.dropout_p < 0.0 || self.dropout_p >= 1.0 {
            return Err(RegError::InvalidProbability(self.dropout_p));
        }
        if self.bn_eps <= 0.0 {
            return Err(RegError::InvalidEpsilon(self.bn_eps));
        }
        if self.bn_momentum < 0.0 || self.bn_momentum > 1.0 {
            return Err(RegError::InvalidMomentum(self.bn_momentum));
        }
        Ok(())
    }
}

/// Configuration settings specifically for Dropout layers.
#[derive(Debug, Clone, PartialEq)]
pub struct DropoutConfig {
    pub p: f64,
    pub in_place: bool,
    pub seed: Option<u64>,
}

impl Default for DropoutConfig {
    fn default() -> Self {
        Self {
            p: 0.5,
            in_place: false,
            seed: None,
        }
    }
}

/// Configuration settings for Normalization layers.
#[derive(Debug, Clone, PartialEq)]
pub struct NormConfig {
    pub eps: f64,
    pub momentum: f64,
    pub affine: bool,
    pub track_running_stats: bool,
}

impl Default for NormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        }
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
