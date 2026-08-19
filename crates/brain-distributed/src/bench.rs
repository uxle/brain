//! # Communication Bandwidth & Latency Benchmarks
//!
//! Benchmarks bus bandwidth and scaling efficiency across collective communication primitives.

/// Collective benchmark suite.
pub struct CommBench {
    pub world_size: usize,
}

impl CommBench {
    /// Creates a new `CommBench`.
    pub fn new(world_size: usize) -> Self {
        Self { world_size }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
