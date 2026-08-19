//! # Core Optimizer Abstractions & Trait
//!
//! Standard interfaces for parameter optimization, gradient steps, parameter groups,
//! zeroing gradients, state serialization, and learning rate access.
#![allow(missing_docs)]

pub mod param_group;

use std::collections::HashMap;
use std::fmt;
use brain_core::Tensor;
pub use param_group::{ParamGroup, ParamId, ParamGroupConfig};

/// Comprehensive error type for optimization operations.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerError {
    EmptyParamGroup,
    GradientDimensionMismatch { expected: Vec<usize>, found: Vec<usize> },
    MissingGradient(ParamId),
    InvalidHyperparameter(String),
    NonFiniteGradient { param_id: ParamId, value: f64 },
    StateDictError(String),
    GroupNotFound(usize),
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::EmptyParamGroup => write!(f, "Parameter group contains no parameters"),
            OptimizerError::GradientDimensionMismatch { expected, found } => {
                write!(f, "Gradient shape {:?} does not match parameter shape {:?}", found, expected)
            }
            OptimizerError::MissingGradient(id) => write!(f, "Missing gradient for parameter {}", id),
            OptimizerError::InvalidHyperparameter(msg) => write!(f, "Invalid hyperparameter: {}", msg),
            OptimizerError::NonFiniteGradient { param_id, value } => {
                write!(f, "Non-finite gradient encountered in param {}: {}", param_id, value)
            }
            OptimizerError::StateDictError(msg) => write!(f, "State dict error: {}", msg),
            OptimizerError::GroupNotFound(idx) => write!(f, "Parameter group index {} out of bounds", idx),
        }
    }
}

impl std::error::Error for OptimizerError {}

pub type OptimResult<T> = Result<T, OptimizerError>;

/// Summary information recorded after an optimizer step.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepInfo {
    pub step_count: usize,
    pub grad_norm: f64,
    pub param_norm: f64,
    pub num_params_updated: usize,
    pub lr_current: f64,
    pub loss_value: Option<f64>,
}

/// Generic optimizer configuration container.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerConfig {
    pub lr: f64,
    pub weight_decay: f64,
    pub eps: f64,
    pub maximize: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            weight_decay: 0.0,
            eps: 1e-8,
            maximize: false,
        }
    }
}

/// Fundamental trait implemented by all first-order and second-order optimizers.
pub trait Optimizer: Send + Sync {
    /// Perform a single optimization step given parameter references and gradients.
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> OptimResult<StepInfo>;

    /// Zero out or reset parameter gradients if stored internally.
    fn zero_grad(&mut self) {}

    /// Returns the active learning rate for the default parameter group.
    fn get_lr(&self) -> f64;

    /// Sets the learning rate for all parameter groups.
    fn set_lr(&mut self, lr: f64);

    /// Sets the learning rate for a specific parameter group index.
    fn set_group_lr(&mut self, group_idx: usize, lr: f64) -> OptimResult<()>;

    /// Returns the number of completed optimizer steps.
    fn get_step_count(&self) -> usize;

    /// Returns immutable slice of parameter groups.
    fn param_groups(&self) -> &[ParamGroup];

    /// Returns mutable slice of parameter groups.
    fn param_groups_mut(&mut self) -> &mut [ParamGroup];

    /// Returns a state dictionary of internal buffers for checkpointing.
    fn state_dict(&self) -> HashMap<String, Tensor>;

    /// Loads internal state from a state dictionary.
    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> OptimResult<()>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
