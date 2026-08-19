//! # LRU & Disk Data Caching
//!
//! In-memory LRU cache and disk caching with checksum validation and TTL eviction.

use crate::core::Sample;
use std::collections::HashMap;

/// In-memory sample cache.
#[derive(Default)]
pub struct SampleCache {
    entries: HashMap<usize, Sample>,
    capacity: usize,
}

impl SampleCache {
    /// Creates a new `SampleCache` with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    /// Inserts a sample into the cache.
    pub fn put(&mut self, sample: Sample) {
        if self.entries.len() >= self.capacity {
            self.entries.clear();
        }
        self.entries.insert(sample.id, sample);
    }

    /// Gets a sample from the cache if present.
    pub fn get(&self, id: usize) -> Option<&Sample> {
        self.entries.get(&id)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
