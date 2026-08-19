//! # Tabular Datasets & Feature Matrices
//!
//! Tabular dataset ingestion from numeric arrays and delimited matrices.

use super::Dataset;
use crate::core::Item;
use brain_core::Tensor;

/// Tabular matrix dataset with labels.
pub struct TabularDataset {
    features: Tensor,
    targets: Option<Tensor>,
}

impl TabularDataset {
    /// Creates a new `TabularDataset` given features matrix `[N, D]`.
    pub fn new(features: Tensor, targets: Option<Tensor>) -> Self {
        Self { features, targets }
    }
}

impl Dataset for TabularDataset {
    fn len(&self) -> usize {
        self.features.shape()[0]
    }

    fn get(&self, idx: usize) -> Option<Item> {
        if idx < self.len() {
            let dim = self.features.shape()[1];
            let feat_data = self.features.data();
            let row_data = feat_data[idx * dim..(idx + 1) * dim].to_vec();
            let row = Tensor::from_vec(row_data, vec![dim]);
            let mut item = Item::new(idx, row);
            if let Some(targets) = &self.targets {
                let target_val = if targets.ndim() == 1 {
                    Tensor::scalar(targets.data()[idx])
                } else {
                    let target_dim = targets.shape()[1];
                    let t_data = targets.data();
                    let t_row = t_data[idx * target_dim..(idx + 1) * target_dim].to_vec();
                    Tensor::from_vec(t_row, vec![target_dim])
                };
                item = item.with_target(target_val);
            }
            Some(item)
        } else {
            None
        }
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
