//! # Cluster Node Management
//!
//! Tracks cluster nodes, IP endpoints, and device assignments.

/// Distributed cluster member node.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub rank: usize,
    pub address: String,
}

impl ClusterNode {
    /// Creates a new `ClusterNode`.
    pub fn new(rank: usize, address: impl Into<String>) -> Self {
        Self {
            rank,
            address: address.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
