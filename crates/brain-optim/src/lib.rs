//! # Brain Optimization Framework (`brain-optim`)
//!
//! Production-grade neural network optimization algorithms, learning rate schedulers,
//! gradient clipping, mixed precision scalability, stochastic weight averaging, and loss analysis.
#![allow(missing_docs)]

pub mod adadelta;
pub mod adafactor;
pub mod adagrad;
pub mod adam;
pub mod adan;
pub mod amp;
pub mod builder;
pub mod clipping;
pub mod ema;
pub mod lamb;
pub mod lion;
pub mod lookahead;
pub mod loss_landscape;
pub mod lr_finder;
pub mod muon;
pub mod novograd;
pub mod optimizer;
pub mod radam;
pub mod rmsprop;
pub mod sam;
pub mod schedulers;
pub mod sgd;
pub mod sophia;
pub mod state;
pub mod swa;

pub use adadelta::{Adadelta, AdadeltaConfig};
pub use adafactor::{Adafactor, AdafactorConfig};
pub use adagrad::{Adagrad, AdagradConfig};
pub use adam::{
    variants::{AdamVariant, Adamax, Nadam},
    Adam, AdamConfig, AdamWConfig,
};
pub use adan::{Adan, AdanConfig};
pub use amp::{AmpConfig, GradScaler};
pub use builder::{OptimizerBuilder, OptimizerKind};
pub use clipping::{
    clip_grad_adaptive_, clip_grad_norm_, clip_grad_value_, GradClipper, NormType, AGC,
};
pub use ema::{EmaConfig, ModelEma};
pub use lamb::{Lamb, LambConfig};
pub use lion::{Lion, LionConfig};
pub use lookahead::{Lookahead, LookaheadConfig};
pub use loss_landscape::{create_filter_normalized_direction, interpolate_1d, LossLandscapeConfig};
pub use lr_finder::{LrFindConfig, LrFindResult, LrFinder};
pub use muon::{Muon, MuonConfig};
pub use novograd::{NovoGrad, NovoGradConfig};
pub use optimizer::{
    OptimResult, Optimizer, OptimizerConfig, OptimizerError, ParamGroup, ParamId, StepInfo,
};
pub use radam::{RAdam, RAdamConfig};
pub use rmsprop::{Rmsprop, RmspropConfig};
pub use sam::{Sam, SamConfig};
pub use schedulers::{
    ConstantWarmup, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR, ExponentialLR,
    ExponentialWarmup, LinearWarmup, LrScheduler, MultiStepLR, OneCycleLR, PolynomialLR,
    ReduceLROnPlateau, StepLR, StepMode,
};
pub use sgd::{nesterov::SgdNesterov, Sgd, SgdConfig};
pub use sophia::{SophiaG, SophiaGConfig};
pub use state::{OptimizerCheckpoint, StateDict};
pub use swa::{SwAConfig, SwAOptimizer};

/// Semantic version of the `brain-optim` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and structs.
pub mod prelude {
    pub use super::adam::{Adam, AdamConfig, AdamWConfig};
    pub use super::amp::GradScaler;
    pub use super::builder::OptimizerBuilder;
    pub use super::clipping::{clip_grad_norm_, clip_grad_value_};
    pub use super::ema::ModelEma;
    pub use super::lamb::{Lamb, LambConfig};
    pub use super::lion::{Lion, LionConfig};
    pub use super::optimizer::{Optimizer, ParamGroup, StepInfo};
    pub use super::rmsprop::{Rmsprop, RmspropConfig};
    pub use super::schedulers::LrScheduler;
    pub use super::sgd::{Sgd, SgdConfig};
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
