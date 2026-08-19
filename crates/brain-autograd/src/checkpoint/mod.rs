//! # Gradient Checkpointing Subsystem
//!
//! Trades computation for memory by discarding intermediate forward activations
//! and recomputing them on-demand during the reverse sweep.

pub mod selective;
pub mod offload;
pub mod cpu_offload;

pub use selective::{checkpoint, CheckpointPolicy};
pub use offload::RecomputeGraph;
pub use cpu_offload::CpuOffloader;

/// Optimal checkpoint schedule calculator (Chen et al. sqrt(N) allocation).
#[derive(Debug, Clone)]
pub struct BudgetCheckpoint {
    pub total_layers: usize,
    pub memory_budget_bytes: usize,
}

impl BudgetCheckpoint {
    /// Creates a budget planner for `total_layers` layers.
    pub fn new(total_layers: usize, memory_budget_bytes: usize) -> Self {
        Self {
            total_layers,
            memory_budget_bytes,
        }
    }

    /// Returns boolean mask indicating which layer indices should be checkpointed.
    pub fn compute_checkpoint_mask(&self) -> Vec<bool> {
        if self.total_layers == 0 {
            return Vec::new();
        }
        let step = (self.total_layers as f64).sqrt().ceil() as usize;
        let step = step.max(1);
        (0..self.total_layers).map(|i| i % step == 0).collect()
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
}
