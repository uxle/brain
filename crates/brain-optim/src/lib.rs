//! # Brain Optimization Framework (`brain-optim`)
//!
//! Production-grade neural network optimization algorithms, learning rate schedulers,
//! gradient clipping, mixed precision scalability, stochastic weight averaging, and loss analysis.
#![allow(missing_docs)]

pub mod optimizer;
pub mod sgd;
pub mod rmsprop;
pub mod adagrad;
pub mod adadelta;
pub mod adam;
pub mod radam;
pub mod lamb;
pub mod lion;
pub mod novograd;
pub mod schedulers;
pub mod clipping;
pub mod lr_finder;
pub mod swa;
pub mod amp;
pub mod ema;
pub mod lookahead;
pub mod sam;
pub mod loss_landscape;
pub mod state;
pub mod builder;

pub use optimizer::{Optimizer, OptimizerConfig, OptimizerError, OptimResult, StepInfo, ParamGroup, ParamId};
pub use sgd::{Sgd, SgdConfig, nesterov::SgdNesterov};
pub use rmsprop::{Rmsprop, RmspropConfig};
pub use adagrad::{Adagrad, AdagradConfig};
pub use adadelta::{Adadelta, AdadeltaConfig};
pub use adam::{Adam, AdamConfig, AdamWConfig, variants::{Adamax, Nadam, AdamVariant}};
pub use radam::{RAdam, RAdamConfig};
pub use lamb::{Lamb, LambConfig};
pub use lion::{Lion, LionConfig};
pub use novograd::{NovoGrad, NovoGradConfig};
pub use schedulers::{
    LrScheduler, StepMode, StepLR, MultiStepLR, ExponentialLR, PolynomialLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR,
    LinearWarmup, ConstantWarmup, ExponentialWarmup, ReduceLROnPlateau,
};
pub use clipping::{GradClipper, clip_grad_norm_, clip_grad_value_, clip_grad_adaptive_, NormType, AGC};
pub use lr_finder::{LrFinder, LrFindConfig, LrFindResult};
pub use swa::{SwAOptimizer, SwAConfig};
pub use amp::{GradScaler, AmpConfig};
pub use ema::{ModelEma, EmaConfig};
pub use lookahead::{Lookahead, LookaheadConfig};
pub use sam::{Sam, SamConfig};
pub use loss_landscape::{interpolate_1d, create_filter_normalized_direction, LossLandscapeConfig};
pub use state::{StateDict, OptimizerCheckpoint};
pub use builder::{OptimizerBuilder, OptimizerKind};

/// Semantic version of the `brain-optim` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and structs.
pub mod prelude {
    pub use super::optimizer::{Optimizer, ParamGroup, StepInfo};
    pub use super::sgd::{Sgd, SgdConfig};
    pub use super::adam::{Adam, AdamConfig, AdamWConfig};
    pub use super::lion::{Lion, LionConfig};
    pub use super::lamb::{Lamb, LambConfig};
    pub use super::rmsprop::{Rmsprop, RmspropConfig};
    pub use super::schedulers::LrScheduler;
    pub use super::clipping::{clip_grad_norm_, clip_grad_value_};
    pub use super::amp::GradScaler;
    pub use super::ema::ModelEma;
    pub use super::builder::OptimizerBuilder;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
