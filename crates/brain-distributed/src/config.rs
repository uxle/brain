//! # Distributed Cluster Configuration
//!
//! Controls communication backends, collective algorithms, timeout durations, and process counts.

/// Supported communication transport backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    #[default]
    Memory,
    Tcp,
}

/// Complete configuration options for a distributed cluster node.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub world_size: usize,
    pub rank: usize,
    pub backend: BackendKind,
    pub timeout_ms: u64,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            backend: BackendKind::default(),
            timeout_ms: 30000,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
