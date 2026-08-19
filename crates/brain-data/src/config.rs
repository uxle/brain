//! # Global Pipeline Configuration
//!
//! Controls thread pool concurrency, batch sizing defaults, and device pin settings.

/// High-level DataLoader configuration.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub shuffle: bool,
    pub drop_last: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: 4,
            shuffle: false,
            drop_last: false,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
