//! # Parallel Processing Configuration
//!
//! Multi-threaded transform processing configuration options.

/// Parallel process configuration.
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    pub thread_count: usize,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self { thread_count: 4 }
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
