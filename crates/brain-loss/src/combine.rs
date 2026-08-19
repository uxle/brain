//! # Loss Composition & Multi-Task Scheduling
//!
//! Weighted linear combination, multiplication, and maximum over multiple loss terms.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::LossResult;

/// Combination mode for multiple loss terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CombineMode {
    #[default]
    WeightedSum,
    Product,
    Max,
}

/// Composite loss orchestrator combining multiple objectives.
#[derive(Debug, Clone, Default)]
pub struct CompositeLoss {
    pub weights: Vec<f64>,
    pub mode: CombineMode,
}

impl CompositeLoss {
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights, mode: CombineMode::WeightedSum }
    }

    pub fn combine(&self, loss_values: &[Tensor]) -> LossResult<Tensor> {
        let n = loss_values.len().min(self.weights.len());
        if n == 0 { return Ok(Tensor::zeros(vec![1])); }

        match self.mode {
            CombineMode::WeightedSum => {
                let mut total = 0.0f64;
                for (i, loss_val) in loss_values.iter().enumerate().take(n) {
                    let v = loss_val.to_vec()[0];
                    total += self.weights[i] * v;
                }
                Ok(Tensor::from_vec(vec![total], vec![1]))
            }
            CombineMode::Product => {
                let mut prod = 1.0f64;
                for loss_val in loss_values.iter().take(n) {
                    let v = loss_val.to_vec()[0];
                    prod *= v;
                }
                Ok(Tensor::from_vec(vec![prod], vec![1]))
            }
            CombineMode::Max => {
                let mut max_v = f64::NEG_INFINITY;
                for (i, loss_val) in loss_values.iter().enumerate().take(n) {
                    let v = loss_val.to_vec()[0] * self.weights[i];
                    if v > max_v { max_v = v; }
                }
                Ok(Tensor::from_vec(vec![max_v], vec![1]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
