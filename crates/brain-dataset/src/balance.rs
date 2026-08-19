//! # Class Rebalancing & Sampling
//!
//! Oversampling, undersampling, and class balancing strategies for skewed datasets.

/// Class rebalancing configuration.
#[derive(Debug, Clone)]
pub struct BalanceConfig {
    pub target_samples_per_class: usize,
}

impl BalanceConfig {
    /// Creates a new `BalanceConfig`.
    pub fn new(target_samples_per_class: usize) -> Self {
        Self { target_samples_per_class }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
