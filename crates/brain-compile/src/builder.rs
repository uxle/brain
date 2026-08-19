//! # Fluent IR Graph Builder with Checkpointing
//!
//! Provides progressive building of IR graphs with speculative checkpoint/rollback capabilities.

use crate::ir::{IrGraph, IrType, OpKind};

/// Progressive builder with checkpoint and rollback support.
pub struct IrGraphBuilder {
    graph: IrGraph,
    snapshots: Vec<IrGraph>,
}

impl Default for IrGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IrGraphBuilder {
    /// Creates a new `IrGraphBuilder`.
    pub fn new() -> Self {
        Self {
            graph: IrGraph::new(),
            snapshots: Vec::new(),
        }
    }

    /// Adds an input value to the graph.
    pub fn add_input(&mut self, dtype: IrType, shape: Vec<usize>) -> usize {
        let id = self.graph.add_value(dtype, shape);
        self.graph.inputs.push(id);
        id
    }

    /// Adds an arithmetic binary addition node.
    pub fn add(&mut self, lhs: usize, rhs: usize, shape: Vec<usize>) -> usize {
        let out = self.graph.add_value(IrType::F64, shape);
        self.graph.add_node(OpKind::Add, vec![lhs, rhs], out);
        out
    }

    /// Creates a checkpoint snapshot of the current graph.
    pub fn checkpoint(&mut self) {
        self.snapshots.push(self.graph.clone());
    }

    /// Rolls back graph state to the latest checkpoint.
    pub fn rollback(&mut self) -> bool {
        if let Some(prev) = self.snapshots.pop() {
            self.graph = prev;
            true
        } else {
            false
        }
    }

    /// Finishes building and consumes the builder.
    pub fn finish(self) -> IrGraph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
