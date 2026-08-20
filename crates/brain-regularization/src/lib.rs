//! # Brain Regularization Framework (`brain-regularization`)
//!
//! Production-grade ML regularization toolkit: Dropout family, Normalization layers,
//! explicit penalty regularizers (L1/L2/ElasticNet), Weight Decay, Early Stopping, and Data Augmentations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

pub mod core;
pub mod ewc;
pub mod config;
pub mod utils;
pub mod dropout;
pub mod normalization;
pub mod regularizers;
pub mod decay;
pub mod earlystop;
pub mod stopping;
pub mod augment;
pub mod perturb;
pub mod dropout_uncertainty;
pub mod label_smooth;
pub mod curriculum;
pub mod consistency;
pub mod rules;
pub mod registry;
pub mod train_hooks;
pub mod ops;
pub mod r#impl;

pub use core::{RegError, RegKind, RegResult, RegState, Regularization};
pub use config::{DropoutConfig, NormConfig, RegConfig};
pub use utils::{XorShift64, WelfordAccumulator, update_ema};
pub use dropout::{Dropout, Dropout2d, alpha::AlphaDropout, adaptive::ConcreteDropout};
pub use normalization::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, BatchNormConfig,
    LayerNorm, LayerNormConfig, RMSNorm,
    GroupNorm, GroupNormConfig, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, InstanceNormConfig,
    SpectralNorm, SpectralNormConfig, WeightNorm,
};
pub use regularizers::{ElasticNetRegularizer, L1Regularizer, L2Regularizer, Regularizer, RegularizerConfig};
pub use decay::{DecayConfig, DecoupledWeightDecay};
pub use earlystop::{EarlyStopConfig, EarlyStopState, EarlyStopping, MetricMode};
pub use stopping::{StopAction, StopOnBudget, StopOnPlateau, StopPolicy};
pub use augment::{Cutout, ImplicitRegConfig, Mixup};
pub use perturb::{GaussianNoise, PerturbConfig, apply_fgsm_perturbation};
pub use dropout_uncertainty::{McDropoutConfig, McDropoutResult, compute_mc_dropout_statistics};
pub use label_smooth::{LabelSmoothConfig, LabelSmoothing};
pub use curriculum::{CurriculumConfig, CurriculumScheduler};
pub use consistency::{ConsistencyConfig, compute_consistency_loss};
pub use rules::{RegStack, WeightedRegularizer};
pub use registry::RegRegistry;
pub use train_hooks::{HookConfig, RegHook};
pub use ops::{dropout_apply, norm_apply_affine};
pub use r#impl::{apply_dropout, apply_layernorm};

/// Semantic version of the `brain-regularization` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and layers.
pub mod prelude {
    pub use super::core::{RegError, RegKind, RegResult, Regularization};
    pub use super::config::RegConfig;
    pub use super::dropout::{Dropout, Dropout2d, alpha::AlphaDropout, adaptive::ConcreteDropout};
    pub use super::normalization::{BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm1d, LayerNorm, RMSNorm, SpectralNorm, WeightNorm};
    pub use super::regularizers::{ElasticNetRegularizer, L1Regularizer, L2Regularizer, Regularizer};
    pub use super::decay::DecoupledWeightDecay;
    pub use super::earlystop::{EarlyStopConfig, EarlyStopping, MetricMode};
    pub use super::augment::Mixup;
    pub use super::rules::RegStack;
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
