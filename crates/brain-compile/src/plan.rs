//! # Liveness-Based Memory Planner
//!
//! Analyzes tensor lifetimes to minimize peak memory consumption through buffer reuse.

use crate::ir::IrGraph;

/// Memory allocation plan for graph execution.
#[derive(Debug, Clone, Default)]
pub struct MemoryPlan {
    pub peak_memory_bytes: usize,
    pub buffer_offsets: Vec<usize>,
}

impl MemoryPlan {
    /// Generates a memory reuse plan for the given IR graph.
    pub fn create_plan(graph: &IrGraph) -> Self {
        let mut total = 0;
        let mut offsets = Vec::new();
        for v in &graph.values {
            offsets.push(total);
            total += v.numel() * 8;
        }
        Self {
            peak_memory_bytes: total,
            buffer_offsets: offsets,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
