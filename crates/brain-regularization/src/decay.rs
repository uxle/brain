//! # Weight Decay Policies
//!
//! Decoupled weight decay (AdamW/SGDW) and L2-equivalent weight shrinkage policies.
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

/// Configuration for weight decay policies.
#[derive(Debug, Clone, PartialEq)]
pub struct DecayConfig {
    pub rate: f64,
    pub decoupled: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            rate: 1e-4,
            decoupled: true,
        }
    }
}

/// Decoupled weight decay executor applying direct parameter shrinkage.
#[derive(Debug, Clone)]
pub struct DecoupledWeightDecay {
    pub rate: f64,
}

impl DecoupledWeightDecay {
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.max(0.0),
        }
    }

    /// Applies decoupled shrinkage in-place given learning rate lr.
    pub fn apply_decay(&self, param: &mut Tensor, lr: f64) {
        let factor = 1.0 - lr * self.rate;
        for val in param.data_mut() {
            *val *= factor;
        }
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
