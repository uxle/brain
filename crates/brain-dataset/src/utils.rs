//! # Dataset Helper Utilities
//!
//! Random number generation and deterministic hashing for dataset indices.

/// Deterministic pseudo-random sequence generator.
pub struct DatasetRng {
    state: u64,
}

impl DatasetRng {
    /// Creates a new `DatasetRng` with seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e3779b97f4a7c15),
        }
    }

    /// Returns next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
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
