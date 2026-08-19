//! # Dataset Registry & Management
//!
//! Register and look up named datasets.

use std::collections::HashMap;

/// Dataset registry storing registered metadata.
#[derive(Default)]
pub struct DatasetRegistry {
    entries: HashMap<String, usize>,
}

impl DatasetRegistry {
    /// Creates a new `DatasetRegistry`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a named dataset with its size.
    pub fn register(&mut self, name: impl Into<String>, size: usize) {
        self.entries.insert(name.into(), size);
    }

    /// Looks up a dataset size by name.
    pub fn lookup(&self, name: &str) -> Option<usize> {
        self.entries.get(name).copied()
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
