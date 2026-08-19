//! # Binary & 2-ary Tree Topology
//!
//! Tree structure for latency-optimal log(N) reduction and broadcast operations.

/// Binary tree communication topology.
#[derive(Debug, Clone)]
pub struct TreeTopology {
    pub rank: usize,
    pub world_size: usize,
}

impl TreeTopology {
    /// Creates a new `TreeTopology`.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size: world_size.max(1),
        }
    }

    /// Returns the parent rank in the tree if not root.
    pub fn parent(&self) -> Option<usize> {
        if self.rank == 0 {
            None
        } else {
            Some((self.rank - 1) / 2)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
