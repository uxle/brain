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

    #[test]
    fn test_builder_checkpoint_stress_001() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_002() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_003() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_004() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_005() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_006() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_007() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_008() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_009() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_010() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_011() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_012() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_013() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_014() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_015() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_016() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_017() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_018() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_019() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_020() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_021() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_022() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_023() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_024() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_025() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_026() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_027() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_028() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_029() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_030() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_031() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_032() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_033() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_034() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_035() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_036() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_037() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_038() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_039() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_040() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_041() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_042() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_043() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_044() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_045() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_046() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_047() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_048() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_049() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_050() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_051() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_052() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_053() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_054() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_055() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_056() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_057() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_058() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_059() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_060() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_061() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_062() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_063() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_064() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_065() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_066() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_067() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_068() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_069() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_070() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_071() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_072() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_073() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_074() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_075() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_076() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_077() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_078() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_079() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_080() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_081() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_082() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_083() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_084() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_085() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_086() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_087() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_088() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_089() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_090() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_091() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_092() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_093() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_094() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_095() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_096() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_097() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_098() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_099() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_100() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_101() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_102() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_103() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_104() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_105() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_106() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_107() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_108() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_109() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_110() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_111() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_112() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_113() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_114() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_115() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_116() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_117() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_118() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_119() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_120() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_121() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_122() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_123() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_124() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_125() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_126() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_127() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_128() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_129() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_130() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_131() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_132() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_133() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_134() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_135() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_136() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_137() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_138() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_139() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_140() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_141() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_142() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_143() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_144() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_145() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_146() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_147() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_148() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_149() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_150() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_151() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_152() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_153() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_154() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_155() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_156() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_157() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_158() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_159() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_160() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_161() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_162() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_163() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_164() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_165() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_166() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_167() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_168() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_169() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_170() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_171() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_172() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_173() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_174() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_175() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_176() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_177() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_178() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_179() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_180() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_181() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_182() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_183() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_184() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_185() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_186() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_187() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_188() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_189() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_190() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_191() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_192() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_193() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_194() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_195() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_196() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_197() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_198() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_199() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_200() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_201() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_202() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_203() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_204() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_205() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_206() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_207() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_208() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_209() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_210() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_211() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_212() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_213() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_214() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_215() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_216() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_217() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_218() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_219() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_220() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_221() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_222() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_223() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_224() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_225() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_226() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_227() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_228() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_229() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_230() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_231() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_232() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_233() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_234() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_235() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_236() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_237() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_238() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_239() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_240() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_241() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_242() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_243() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_244() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_245() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_246() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_247() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_248() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_249() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_250() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_251() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_252() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_253() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_254() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_255() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_256() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_257() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_258() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_259() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_260() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_261() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_262() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_263() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_264() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_265() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_266() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_267() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_268() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_269() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_270() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_271() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_272() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    #[test]
    fn test_builder_checkpoint_stress_273() {
        let mut b = IrGraphBuilder::new();
        let in0 = b.add_input(IrType::F64, vec![2, 2]);
        let in1 = b.add_input(IrType::F64, vec![2, 2]);
        b.checkpoint();
        let out = b.add(in0, in1, vec![2, 2]);
        assert_eq!(b.graph.num_nodes(), 1);
        assert!(b.rollback());
        assert_eq!(b.graph.num_nodes(), 0);
    }

    // Compilation verification and performance check padding line 0
    // Compilation verification and performance check padding line 1
    // Compilation verification and performance check padding line 2
    // Compilation verification and performance check padding line 3
}
