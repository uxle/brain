//! # Concurrent Data Loaders
//!
//! Abstract data loading interfaces for memory-resident and disk-backed dataset sources.

use crate::core::{DataSource, Sample};
use brain_core::Tensor;

/// In-memory dataset loader.
pub struct MemoryLoader {
    samples: Vec<Sample>,
}

impl MemoryLoader {
    /// Creates a new `MemoryLoader` from a list of tensors.
    pub fn from_tensors(tensors: Vec<Tensor>) -> Self {
        let samples = tensors
            .into_iter()
            .enumerate()
            .map(|(i, t)| Sample::new(i, t))
            .collect();
        Self { samples }
    }
}

impl DataSource for MemoryLoader {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Option<Sample> {
        self.samples.get(idx).cloned()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
