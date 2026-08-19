//! # Core Distributed Types & Process Context
//!
//! Provides the primary [`DistributedContext`], [`Rank`], and [`WorldSize`] abstractions.

/// Unique identifier of a process in a distributed cluster.
pub type Rank = usize;

/// Total number of processes participating in a distributed cluster.
pub type WorldSize = usize;

/// Complete execution context for a distributed process node.
#[derive(Debug, Clone)]
pub struct DistributedContext {
    pub rank: Rank,
    pub world_size: WorldSize,
    pub local_rank: usize,
}

impl DistributedContext {
    /// Creates a new `DistributedContext`.
    pub fn new(rank: Rank, world_size: WorldSize) -> Self {
        Self {
            rank,
            world_size: world_size.max(1),
            local_rank: rank % world_size.max(1),
        }
    }

    /// Returns whether this node is the master coordinator (rank 0).
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
