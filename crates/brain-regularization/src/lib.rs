//! # Brain Regularization Framework (`brain-regularization`)
//!
//! Production-grade ML regularization toolkit: Dropout family, Normalization layers,
//! explicit penalty regularizers (L1/L2/ElasticNet), Weight Decay, Early Stopping, and Data Augmentations.
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

pub mod augment;
pub mod config;
pub mod consistency;
pub mod core;
pub mod curriculum;
pub mod decay;
pub mod dropout;
pub mod dropout_uncertainty;
pub mod earlystop;
pub mod ewc;
pub mod r#impl;
pub mod label_smooth;
pub mod normalization;
pub mod ops;
pub mod perturb;
pub mod registry;
pub mod regularizers;
pub mod rules;
pub mod stopping;
pub mod train_hooks;
pub mod utils;

pub use augment::{Cutout, ImplicitRegConfig, Mixup};
pub use config::{DropoutConfig, NormConfig, RegConfig};
pub use consistency::{compute_consistency_loss, ConsistencyConfig};
pub use core::{RegError, RegKind, RegResult, RegState, Regularization};
pub use curriculum::{CurriculumConfig, CurriculumScheduler};
pub use decay::{DecayConfig, DecoupledWeightDecay};
pub use dropout::{adaptive::ConcreteDropout, alpha::AlphaDropout, Dropout, Dropout2d};
pub use dropout_uncertainty::{compute_mc_dropout_statistics, McDropoutConfig, McDropoutResult};
pub use earlystop::{EarlyStopConfig, EarlyStopState, EarlyStopping, MetricMode};
pub use label_smooth::{LabelSmoothConfig, LabelSmoothing};
pub use normalization::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, BatchNormConfig, GroupNorm, GroupNormConfig,
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, InstanceNormConfig, LayerNorm, LayerNormConfig,
    RMSNorm, SpectralNorm, SpectralNormConfig, WeightNorm,
};
pub use ops::{dropout_apply, norm_apply_affine};
pub use perturb::{apply_fgsm_perturbation, GaussianNoise, PerturbConfig};
pub use r#impl::{apply_dropout, apply_layernorm};
pub use registry::RegRegistry;
pub use regularizers::{
    ElasticNetRegularizer, L1Regularizer, L2Regularizer, Regularizer, RegularizerConfig,
};
pub use rules::{RegStack, WeightedRegularizer};
pub use stopping::{StopAction, StopOnBudget, StopOnPlateau, StopPolicy};
pub use train_hooks::{HookConfig, RegHook};
pub use utils::{update_ema, WelfordAccumulator, XorShift64};

/// Semantic version of the `brain-regularization` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and layers.
pub mod prelude {
    pub use super::augment::Mixup;
    pub use super::config::RegConfig;
    pub use super::core::{RegError, RegKind, RegResult, Regularization};
    pub use super::decay::DecoupledWeightDecay;
    pub use super::dropout::{adaptive::ConcreteDropout, alpha::AlphaDropout, Dropout, Dropout2d};
    pub use super::earlystop::{EarlyStopConfig, EarlyStopping, MetricMode};
    pub use super::normalization::{
        BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm1d, LayerNorm, RMSNorm, SpectralNorm,
        WeightNorm,
    };
    pub use super::regularizers::{
        ElasticNetRegularizer, L1Regularizer, L2Regularizer, Regularizer,
    };
    pub use super::rules::RegStack;
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
