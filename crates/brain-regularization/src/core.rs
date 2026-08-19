//! # Core Regularization Traits & Representations
//!
//! Universal regularization traits, layer categories, error handling, and state management.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

use std::fmt;
use brain_core::Tensor;

/// Distinct categories of regularization techniques.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegKind {
    #[default]
    Dropout,
    AlphaDropout,
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    WeightNorm,
    SpectralNorm,
    L1,
    L2,
    ElasticNet,
    EarlyStopping,
    Mixup,
    LabelSmoothing,
}

/// State container for tracking regularization dynamics during training.
#[derive(Debug, Clone, PartialEq)]
pub struct RegState {
    pub is_training: bool,
    pub step_count: usize,
    pub epoch_count: usize,
    pub total_penalty: f64,
}

impl Default for RegState {
    fn default() -> Self {
        Self {
            is_training: true,
            step_count: 0,
            epoch_count: 0,
            total_penalty: 0.0,
        }
    }
}

/// Fundamental trait implemented by all regularization transforms.
pub trait Regularization: Send + Sync {
    /// Applies regularization transformation to an input tensor.
    fn apply(&mut self, input: &Tensor) -> RegResult<Tensor>;

    /// Sets the layer to training mode.
    fn train_mode(&mut self) {}

    /// Sets the layer to evaluation / inference mode.
    fn eval_mode(&mut self) {}

    /// Computes explicit loss penalty contribution (e.g. for L1/L2 weight decay).
    fn compute_penalty(&self, _params: &[Tensor]) -> f64 {
        0.0
    }

    /// Returns the architectural kind of regularization.
    fn kind(&self) -> RegKind;
}

/// Comprehensive error type for regularization failures.
#[derive(Debug, Clone, PartialEq)]
pub enum RegError {
    InvalidProbability(f64),
    InvalidEpsilon(f64),
    InvalidMomentum(f64),
    ShapeMismatch { expected: Vec<usize>, found: Vec<usize> },
    EmptyTensor,
    InvalidGroupCount { num_groups: usize, num_channels: usize },
    EarlyStopTriggered { epoch: usize, metric: f64 },
    ConfigurationError(String),
}

impl fmt::Display for RegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegError::InvalidProbability(p) => write!(f, "Invalid probability factor: {} (must be in [0, 1])", p),
            RegError::InvalidEpsilon(eps) => write!(f, "Invalid epsilon numerical stabilizer: {} (must be > 0)", eps),
            RegError::InvalidMomentum(m) => write!(f, "Invalid momentum factor: {} (must be in [0, 1])", m),
            RegError::ShapeMismatch { expected, found } => {
                write!(f, "Shape mismatch: expected {:?}, found {:?}", expected, found)
            }
            RegError::EmptyTensor => write!(f, "Cannot apply regularization on empty tensor"),
            RegError::InvalidGroupCount { num_groups, num_channels } => {
                write!(f, "Channels {} not divisible by group count {}", num_channels, num_groups)
            }
            RegError::EarlyStopTriggered { epoch, metric } => {
                write!(f, "Early stopping triggered at epoch {} with metric value {}", epoch, metric)
            }
            RegError::ConfigurationError(msg) => write!(f, "Regularization configuration error: {}", msg),
        }
    }
}

impl std::error::Error for RegError {}

pub type RegResult<T> = Result<T, RegError>;

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
