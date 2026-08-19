//! # Artifact & Command Result Caching
//!
//! Stores parsed datasets, model weights, and benchmark baselines with key-value invalidation.

use std::collections::HashMap;

/// In-memory and disk cache store for CLI operations.
#[derive(Default)]
pub struct CliCache {
    entries: HashMap<String, Vec<u8>>,
}

impl CliCache {
    /// Creates a new `CliCache`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Puts a binary entry into the cache.
    pub fn put(&mut self, key: impl Into<String>, data: Vec<u8>) {
        self.entries.insert(key.into(), data);
    }

    /// Retrieves an entry from the cache.
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.entries.get(key).map(|v| v.as_slice())
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
