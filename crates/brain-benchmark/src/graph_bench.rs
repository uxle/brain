//! # Computation Graph & Autodiff Overhead Benchmarks
//!
//! Measures latency of building dynamic execution graphs, topological sweeps,
//! activation tracking, and gradient checkpointing recomputation.

use crate::core::{BenchConfig, BenchResult};
use crate::runner::Runner;
use brain_core::BrainResult;

/// Benchmarks the allocation and tracking overhead of dynamic graph nodes.
pub fn bench_graph_node_overhead(num_nodes: usize) -> BrainResult<BenchResult> {
    let bench_cfg = BenchConfig::new(format!("graph_node_build_{}", num_nodes))
        .with_tag("graph")
        .with_tag("overhead");

    Runner::run_benchmark(&bench_cfg, || {
        let mut nodes = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            nodes.push((i, i + 1));
        }
        std::hint::black_box(nodes);
    })
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
