//! # Contrastive & Metric Representation Losses
//!
//! InfoNCE, Triplet margin loss, SimCLR / NT-Xent representation learning.
#![allow(missing_docs)]

pub mod infonce;
pub mod triplet;
pub mod simclr;

pub use infonce::{InfoNCELoss, InfoNceConfig};
pub use triplet::{TripletMarginLoss, TripletConfig};
pub use simclr::{SimCLRLoss, SimclrConfig};

use brain_core::Tensor;
use crate::core::LossResult;

/// Configuration for contrastive loss modules.
#[derive(Debug, Clone)]
pub struct ContrastiveConfig {
    pub temperature: f64,
    pub margin: f64,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self { temperature: 0.07, margin: 1.0 }
    }
}

/// Trait for self-supervised and pair-based contrastive loss objectives.
pub trait ContrastiveLoss: Send + Sync {
    /// Computes contrastive loss between queries, positive keys, and negative keys.
    fn compute(&self, queries: &Tensor, pos_keys: &Tensor, neg_keys: &[Tensor]) -> LossResult<Tensor>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
