//! # brain-loss
//!
//! Numerically bulletproof loss functions for the Brain Deep Learning Framework:
//! classification, regression, contrastive, adversarial, metric learning,
//! segmentation, knowledge distillation, and composite multi-task scheduling.
//!
//! ## Architecture
//! - [`classification`] — CrossEntropy, Focal, Hinge, KL Divergence
//! - [`regression`] — MSE, MAE, Huber, Smooth L1, Quantile, Cauchy, CosineEmbedding
//! - [`contrastive`] — InfoNCE, Triplet Margin, SimCLR (NT-Xent)
//! - [`adversarial`] — WGAN, Hinge Adversarial, LSGAN, Relativistic GAN
//! - [`segmentation`] — Combined Cross-Entropy + Soft Dice Loss
//! - [`metric_loss`] — ArcFace additive angular margin loss
//! - [`distillation`] — Temperature-scaled soft target knowledge distillation
//! - [`masked`] — Masked loss wrappers and padding-aware reductions
//! - [`combine`] — Composite loss orchestrator (weighted sum, product, max)
//! - [`config`] — `LossConfig`, validation, and hyperparameter descriptors
//! - [`core`] — `Loss` trait, `LossKind`, `LossValue`, `Reduction`, `LossError`
//! - [`ops`] — Fused log-softmax, softmax, NLL, and one-hot encoding
//! - [`utils`] — Reduction helpers, shape verification, and numerical clamping

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod adversarial;
pub mod classification;
pub mod combine;
pub mod config;
pub mod contrastive;
pub mod core;
pub mod distillation;
pub mod impl_;
pub mod masked;
pub mod metric_loss;
pub mod ops;
pub mod regression;
pub mod segmentation;
pub mod utils;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use adversarial::{
    AdvLossConfig, AdvLossKind, AdversarialLoss, HingeAdversarialLoss, LSGANLoss, RelativisticLoss,
    WassersteinConfig, WassersteinLoss,
};
pub use classification::{
    CTCConfig, CTCLoss, ClassLossConfig, ClassLossKind, ClassificationLoss, CrossEntropyConfig,
    CrossEntropyLoss, FocalConfig, FocalLoss, HingeLoss, KLDivergenceLoss,
};
pub use combine::{CombineMode, CompositeLoss};
pub use config::LossConfig;
pub use contrastive::{
    ContrastiveConfig, ContrastiveLoss, InfoNCELoss, InfoNceConfig, SimCLRLoss, SimclrConfig,
    TripletConfig, TripletMarginLoss,
};
pub use core::{Loss, LossError, LossKind, LossResult, LossValue, Reduction};
pub use distillation::{DistillConfig, KnowledgeDistillationLoss};
pub use impl_::{compute_loss, default_config, loss_name};
pub use masked::apply_loss_mask;
pub use metric_loss::{ArcFaceLoss, MetricConfig};
pub use ops::{log_softmax, log_sum_exp_2d, nll_loss, one_hot_target, softmax};
pub use regression::{
    AngularDistanceLoss, CauchyLoss, CosineEmbeddingLoss, HuberLoss, MAELoss, MSELoss,
    QuantileLoss, RegLossConfig, RegressionLoss, RobustConfig, SmoothL1Loss,
};
pub use segmentation::{CEDiceLoss, SegLossConfig};
pub use utils::{check_shapes, clamp_eps, reduction_apply, weighted_average};

/// Framework version string.
pub const VERSION: &str = "0.2.0";
