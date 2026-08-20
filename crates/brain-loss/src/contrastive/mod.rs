//! # Contrastive & Metric Representation Losses
//!
//! InfoNCE, Triplet margin loss, SimCLR / NT-Xent representation learning.
#![allow(missing_docs)]

pub mod cosine_embedding;
pub mod infonce;
pub mod margin_ranking;
pub mod simclr;
pub mod triplet;

pub use cosine_embedding::CosineEmbeddingLoss;
pub use infonce::{InfoNCELoss, InfoNceConfig};
pub use margin_ranking::MarginRankingLoss;
pub use simclr::{SimCLRLoss, SimclrConfig};
pub use triplet::{TripletConfig, TripletMarginLoss};

use crate::core::LossResult;
use brain_core::Tensor;

/// Configuration for contrastive loss modules.
#[derive(Debug, Clone)]
pub struct ContrastiveConfig {
    pub temperature: f64,
    pub margin: f64,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            margin: 1.0,
        }
    }
}

/// Trait for self-supervised and pair-based contrastive loss objectives.
pub trait ContrastiveLoss: Send + Sync {
    /// Computes contrastive loss between queries, positive keys, and negative keys.
    fn compute(
        &self,
        queries: &Tensor,
        pos_keys: &Tensor,
        neg_keys: &[Tensor],
    ) -> LossResult<Tensor>;
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
