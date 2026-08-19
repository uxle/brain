//! # Dataset Splitting & Cross-Validation
//!
//! Provides train/val/test splits, stratified splitting, and k-fold cross-validation.

/// Container for partitioned index splits.
pub struct SplitResult {
    pub train_indices: Vec<usize>,
    pub val_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

/// Splits dataset indices into train/val/test sets given fractions.
pub fn random_split_indices(total: usize, train_frac: f64, val_frac: f64) -> SplitResult {
    let train_count = ((total as f64) * train_frac) as usize;
    let val_count = ((total as f64) * val_frac) as usize;

    let train_indices = (0..train_count).collect();
    let val_indices = (train_count..train_count + val_count).collect();
    let test_indices = (train_count + val_count..total).collect();

    SplitResult {
        train_indices,
        val_indices,
        test_indices,
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
