//! # Dynamic Regularization Registry
//!
//! Name-based dynamic factory lookup for regularization modules and hyperparameters.
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

use super::core::{RegError, RegKind, RegResult};

/// Dynamic factory registry.
#[derive(Debug, Clone, Default)]
pub struct RegRegistry;

impl RegRegistry {
    /// Resolves regularization kind from human-readable identifier.
    pub fn parse_kind(name: &str) -> RegResult<RegKind> {
        match name.to_lowercase().as_str() {
            "dropout" => Ok(RegKind::Dropout),
            "alpha_dropout" => Ok(RegKind::AlphaDropout),
            "batch_norm" | "batchnorm" => Ok(RegKind::BatchNorm),
            "layer_norm" | "layernorm" => Ok(RegKind::LayerNorm),
            "group_norm" | "groupnorm" => Ok(RegKind::GroupNorm),
            "instance_norm" | "instancenorm" => Ok(RegKind::InstanceNorm),
            "weight_norm" | "weightnorm" => Ok(RegKind::WeightNorm),
            "spectral_norm" | "spectralnorm" => Ok(RegKind::SpectralNorm),
            "l1" => Ok(RegKind::L1),
            "l2" => Ok(RegKind::L2),
            "elastic_net" => Ok(RegKind::ElasticNet),
            _ => Err(RegError::ConfigurationError(format!(
                "Unknown regularization layer: {}",
                name
            ))),
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
