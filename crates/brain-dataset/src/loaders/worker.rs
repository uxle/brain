//! # Asynchronous Worker Pool
//!
//! Spawns background worker threads for non-blocking dataset loading and batch collation.

/// Bounded worker pool.
pub struct WorkerPool {
    pub num_workers: usize,
}

impl WorkerPool {
    /// Creates a new `WorkerPool`.
    pub fn new(num_workers: usize) -> Self {
        Self { num_workers }
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
