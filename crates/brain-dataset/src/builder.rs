//! # Fluent Dataset Builder API
//!
//! Fluent chaining interface for building and configuring complex dataset pipelines.

use crate::config::DatasetConfig;

/// Fluent dataset pipeline builder.
pub struct DatasetBuilder {
    config: DatasetConfig,
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self {
            config: DatasetConfig::default(),
        }
    }
}

impl DatasetBuilder {
    /// Creates a new `DatasetBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Sets whether to shuffle items.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    /// Builds the configured `DatasetConfig`.
    pub fn build(self) -> DatasetConfig {
        self.config
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
