//! # Loss Core Types
//!
//! Loss trait, reduction modes, loss kinds, error types, and loss values.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Reduction mode for aggregating loss across batch elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Reduction {
    #[default]
    Mean,
    Sum,
    None,
}

/// Enumeration of all loss functions supported in `brain-loss`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LossKind {
    #[default]
    CrossEntropy,
    BinaryCrossEntropy,
    Focal,
    Hinge,
    KLDivergence,
    MSE,
    MAE,
    Huber,
    SmoothL1,
    Quantile,
    CosineEmbedding,
    InfoNCE,
    Triplet,
    SimCLR,
    Wasserstein,
    Dice,
    ArcFace,
    KnowledgeDistillation,
}

/// Error type for loss computations.
#[derive(Debug, Clone, PartialEq)]
pub enum LossError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    InvalidTarget(String),
    NumericalInstability(String),
    InvalidParameter(String),
}

impl std::fmt::Display for LossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossError::ShapeMismatch { expected, got } => write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got),
            LossError::InvalidTarget(msg) => write!(f, "Invalid target: {}", msg),
            LossError::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
            LossError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

pub type LossResult<T> = Result<T, LossError>;

/// Represents the evaluated loss value and optional per-sample breakdown.
#[derive(Debug, Clone)]
pub struct LossValue {
    pub scalar: f64,
    pub per_sample: Option<Tensor>,
}

impl LossValue {
    pub fn new(scalar: f64) -> Self {
        Self { scalar, per_sample: None }
    }

    pub fn with_per_sample(scalar: f64, per_sample: Tensor) -> Self {
        Self { scalar, per_sample: Some(per_sample) }
    }
}

use brain_autograd::Value;

/// Unified trait for all loss functions.
pub trait Loss: Send + Sync {
    /// Name of the loss function.
    fn name(&self) -> &'static str;
    /// Evaluates the loss given prediction and ground-truth target tensors.
    fn forward(&self, pred: &Tensor, target: &Tensor) -> LossResult<Tensor>;
    /// Evaluates the differentiable loss given prediction `Value` and ground-truth target tensor.
    fn forward_value(&self, pred: &Value, target: &Tensor) -> LossResult<Value> {
        let t_loss = self.forward(pred.data(), target)?;
        Ok(Value::new(t_loss, pred.requires_grad()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
