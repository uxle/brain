//! # Ring Topology Primitives
//!
//! Construct circular logical topologies for bandwidth-optimal ring allreduce.

/// Ring communication topology for a single rank.
#[derive(Debug, Clone)]
pub struct RingTopology {
    pub rank: usize,
    pub world_size: usize,
}

impl RingTopology {
    /// Creates a new `RingTopology`.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size: world_size.max(1),
        }
    }

    /// Returns the rank of the left neighbor.
    pub fn left_neighbor(&self) -> usize {
        (self.rank + self.world_size - 1) % self.world_size
    }

    /// Returns the rank of the right neighbor.
    pub fn right_neighbor(&self) -> usize {
        (self.rank + 1) % self.world_size
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
