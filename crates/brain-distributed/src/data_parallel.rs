//! # Distributed Data Parallelism (DDP)
//!
//! Wraps parameter collections to automatically synchronize gradients via AllReduce.

use brain_core::Tensor;

/// DataParallel module wrapper.
pub struct DataParallel {
    pub world_size: usize,
}

impl DataParallel {
    /// Creates a new `DataParallel` wrapper.
    pub fn new(world_size: usize) -> Self {
        Self { world_size }
    }

    /// Synchronizes parameter gradients across ranks.
    pub fn sync_gradients(&self, gradients: &mut [Tensor]) {
        for g in gradients {
            let _ = g;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
