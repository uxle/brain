//! # Graph Implementation Helpers
//!
//! Convenient evaluation, execution runners, and batch evaluation wrappers.
#![allow(missing_docs)]

use crate::core::GraphResult;
use crate::interp::GraphInterpreter;
use crate::ir::GraphIr;
use brain_core::Tensor;

/// Runs inference on a `GraphIr` given input tensors.
pub fn run_graph(graph: &GraphIr, inputs: &[Tensor]) -> GraphResult<Vec<Tensor>> {
    let mut interp = GraphInterpreter::new();
    interp.run(graph, inputs)
}

/// Computes the total memory allocated by all tensor outputs of the graph.
pub fn total_output_memory_bytes(graph: &GraphIr) -> usize {
    graph
        .values
        .iter()
        .map(|v| v.shape.num_elements() * 4)
        .sum()
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
