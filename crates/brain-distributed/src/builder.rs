//! # Fluent Distributed Builder API
//!
//! Fluent configuration builder for multi-process distributed environments.

use crate::config::{BackendKind, DistributedConfig};

/// Fluent builder for distributed contexts.
#[derive(Default)]
pub struct DistributedBuilder {
    config: DistributedConfig,
}

impl DistributedBuilder {
    /// Creates a new `DistributedBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the world size.
    pub fn world_size(mut self, size: usize) -> Self {
        self.config.world_size = size;
        self
    }

    /// Sets the local process rank.
    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }

    /// Sets the transport backend.
    pub fn backend(mut self, backend: BackendKind) -> Self {
        self.config.backend = backend;
        self
    }

    /// Builds the `DistributedConfig`.
    pub fn build(self) -> DistributedConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
