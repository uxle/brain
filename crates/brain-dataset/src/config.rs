//! # Dataset Configuration
//!
//! Controls batch sizes, worker counts, and dataset loading parameters.

/// Configuration parameters for dataset pipelines.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub num_workers: usize,
    pub drop_last: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: false,
            num_workers: 0,
            drop_last: false,
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
