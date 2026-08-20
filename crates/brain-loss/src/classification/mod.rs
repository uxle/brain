//! # Classification Losses
//!
//! Classification loss trait and modular loss functions (Cross-Entropy, Focal, Hinge, KL).
#![allow(missing_docs)]

pub mod bce;
pub mod ce;
pub mod focal;
pub mod other;

pub use bce::{BCEConfig, BCELoss, BCEWithLogitsLoss};
pub use ce::{CrossEntropyConfig, CrossEntropyLoss};
pub use focal::{FocalConfig, FocalLoss};
pub use other::{ClassLossKind, HingeLoss, KLDivergenceLoss};

use crate::core::{LossResult, Reduction};
use brain_core::Tensor;

/// Configuration for classification loss algorithms.
#[derive(Debug, Clone)]
pub struct ClassLossConfig {
    pub reduction: Reduction,
    pub label_smoothing: f64,
    pub class_weights: Option<Vec<f64>>,
    pub ignore_index: Option<usize>,
}

impl Default for ClassLossConfig {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            label_smoothing: 0.0,
            class_weights: None,
            ignore_index: None,
        }
    }
}

/// Specialized trait for discrete and multi-label classification loss modules.
pub trait ClassificationLoss: Send + Sync {
    /// Computes classification loss from logits [N, C] and targets [N].
    fn compute(&self, logits: &Tensor, targets: &[usize]) -> LossResult<Tensor>;

    /// Computes differentiable classification loss from logits `Value` and targets [N].
    fn forward_value(
        &self,
        logits: &brain_autograd::Value,
        targets: &[usize],
    ) -> LossResult<brain_autograd::Value> {
        let t_loss = self.compute(logits.data(), targets)?;
        Ok(brain_autograd::Value::new(t_loss, logits.requires_grad()))
    }
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
