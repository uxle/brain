//! # Dataset Iteration Implementation
//!
//! Iteration runners over dataset sources.

use crate::core::Batch;
use crate::dataset::Dataset;

/// Iterates sequentially over a dataset yielding batches.
pub struct DatasetIterator<'a, D: Dataset> {
    dataset: &'a D,
    pos: usize,
    batch_size: usize,
}

impl<'a, D: Dataset> DatasetIterator<'a, D> {
    /// Creates a new `DatasetIterator`.
    pub fn new(dataset: &'a D, batch_size: usize) -> Self {
        Self {
            dataset,
            pos: 0,
            batch_size: batch_size.max(1),
        }
    }
}

impl<'a, D: Dataset> Iterator for DatasetIterator<'a, D> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let total = self.dataset.len();
        if self.pos >= total {
            return None;
        }

        let mut items = Vec::new();
        while self.pos < total && items.len() < self.batch_size {
            if let Some(item) = self.dataset.get(self.pos) {
                items.push(item);
            }
            self.pos += 1;
        }

        if items.is_empty() {
            None
        } else {
            Some(Batch::new(items))
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
