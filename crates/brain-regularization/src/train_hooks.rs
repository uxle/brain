//! # Training Loop Lifecycle Hooks
//!
//! Interceptors for applying weight decay, logging penalties, and tracking early stopping state.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use brain_core::Tensor;

/// Configuration for training lifecycle hooks.
#[derive(Debug, Clone, PartialEq)]
pub struct HookConfig {
    pub enable_weight_decay: bool,
    pub weight_decay_rate: f64,
}

impl Default for HookConfig {
    fn default() -> Self {
        Self {
            enable_weight_decay: true,
            weight_decay_rate: 1e-4,
        }
    }
}

/// Regularization Training Hook intercepting forward/backward training passes.
#[derive(Debug, Clone)]
pub struct RegHook {
    pub config: HookConfig,
}

impl RegHook {
    pub fn new(config: HookConfig) -> Self {
        Self { config }
    }

    /// Hook executed after gradient descent optimizer step to apply decoupled decay.
    pub fn after_optimizer_step(&self, params: &mut [Tensor], lr: f64) {
        if !self.config.enable_weight_decay || self.config.weight_decay_rate == 0.0 {
            return;
        }
        let factor = 1.0 - lr * self.config.weight_decay_rate;
        for p in params.iter_mut() {
            for v in p.data_mut() {
                *v *= factor;
            }
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
