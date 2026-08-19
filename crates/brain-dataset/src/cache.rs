//! # In-Memory & Disk Dataset Caching
//!
//! Caches processed samples to eliminate redundant transform computations across epochs.

use crate::core::Item;
use std::collections::HashMap;

/// In-memory dataset cache.
#[derive(Default)]
pub struct DatasetCache {
    items: HashMap<usize, Item>,
}

impl DatasetCache {
    /// Creates a new `DatasetCache`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Caches an item.
    pub fn insert(&mut self, item: Item) {
        self.items.insert(item.id, item);
    }

    /// Retrieves an item.
    pub fn get(&self, id: usize) -> Option<&Item> {
        self.items.get(&id)
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
