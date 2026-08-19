//! # Streaming Execution Engine
//!
//! Executes compiled computation graphs with memory buffer bindings and deterministic guarantees.

use crate::core::CompilationError;
use crate::ir::IrGraph;
use brain_core::Tensor;

/// Streaming execution engine for IR graphs.
pub struct ExecutionEngine {
    graph: IrGraph,
}

impl ExecutionEngine {
    /// Creates a new `ExecutionEngine` for the given graph.
    pub fn new(graph: IrGraph) -> Self {
        Self { graph }
    }

    /// Executes the graph with the provided input tensors.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, CompilationError> {
        let interp = crate::backend::Interpreter::new();
        interp.evaluate(&self.graph, inputs)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
