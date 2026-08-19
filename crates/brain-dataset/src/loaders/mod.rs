//! # Multi-Worker Data Loaders
//!
//! High-throughput `DataLoader` orchestrating dataset batching, sampling, and worker prefetching.

pub mod worker;

pub use worker::WorkerPool;

use crate::core::Batch;
use crate::dataset::Dataset;

/// High-level DataLoader wrapping a dataset.
pub struct DataLoader<'a, D: Dataset> {
    pub dataset: &'a D,
    pub batch_size: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    /// Creates a new `DataLoader`.
    pub fn new(dataset: &'a D, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size: batch_size.max(1),
        }
    }

    /// Fetches the first batch from the dataset.
    pub fn fetch_batch(&self) -> Option<Batch> {
        let mut items = Vec::new();
        for i in 0..self.batch_size.min(self.dataset.len()) {
            if let Some(it) = self.dataset.get(i) {
                items.push(it);
            }
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
