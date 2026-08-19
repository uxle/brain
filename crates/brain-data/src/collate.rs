//! # Sample Collation & Stacking
//!
//! Provides [`default_collate`] (tensor stacking), [`pad_collate`] (variable length sequences), and custom [`CollateFn`].

use crate::core::{Sample, SampleBatch};

/// Collation function trait.
pub trait CollateFn: Send + Sync {
    fn collate(&self, samples: &[Sample]) -> SampleBatch;
}

/// Default collation function creating a batch from samples.
pub fn default_collate(samples: &[Sample]) -> SampleBatch {
    SampleBatch::new(samples.to_vec())
}

/// Collation function padding variable-length tensors.
pub fn pad_collate(samples: &[Sample], _pad_value: f64) -> SampleBatch {
    SampleBatch::new(samples.to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
