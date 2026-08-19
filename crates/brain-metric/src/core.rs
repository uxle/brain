//! # Metric Core Types
//!
//! Metric trait, MetricValue container, MetricKind identifiers, and error structures.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Value representation produced by metric calculations.
#[derive(Debug, Clone)]
pub enum MetricValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
    Table { headers: Vec<String>, rows: Vec<Vec<String>> },
}

impl MetricValue {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            MetricValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }
}

/// Enumeration of standard metric kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(non_camel_case_types)]
pub enum MetricKind {
    #[default]
    Accuracy,
    TopKAccuracy,
    Precision,
    Recall,
    F1Score,
    ROCAUC,
    PRAUC,
    MSE,
    RMSE,
    MAE,
    MAPE,
    R2Score,
    IoU,
    mAP,
    BLEU,
    ROUGE,
    NDCG,
    Silhouette,
    MCC,
}

/// Error type for metric evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricError {
    LengthMismatch { expected: usize, got: usize },
    InvalidThreshold(String),
    EmptyInput,
    UndefinedMetric(String),
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricError::LengthMismatch { expected, got } => write!(f, "Length mismatch: expected {}, got {}", expected, got),
            MetricError::InvalidThreshold(msg) => write!(f, "Invalid threshold: {}", msg),
            MetricError::EmptyInput => write!(f, "Input evaluation set cannot be empty"),
            MetricError::UndefinedMetric(msg) => write!(f, "Undefined metric: {}", msg),
        }
    }
}

pub type MetricResult<T> = Result<T, MetricError>;

/// Incremental and batch metric evaluation trait.
pub trait Metric: Send + Sync {
    /// Name of the metric.
    fn name(&self) -> &'static str;
    /// Resets accumulator state.
    fn reset(&mut self);
    /// Updates metric with a new batch of predictions and targets.
    fn update(&mut self, preds: &Tensor, targets: &Tensor) -> MetricResult<()>;
    /// Computes and returns the current accumulated metric value.
    fn compute(&self) -> MetricResult<MetricValue>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
