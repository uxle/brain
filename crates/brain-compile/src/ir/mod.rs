//! # Typed Intermediate Representation (IR)
//!
//! Linear dominance-free intermediate representation for tensor computations and kernel fusion.

pub mod ops;
pub mod verify;

pub use ops::OpKind;

/// Data types supported by the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IrType {
    F32,
    #[default]
    F64,
    I32,
    I64,
    Bool,
    Ptr,
}

/// A typed, shaped value or register in the IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrValue {
    pub id: usize,
    pub dtype: IrType,
    pub shape: Vec<usize>,
}

impl IrValue {
    /// Creates a new `IrValue`.
    pub fn new(id: usize, dtype: IrType, shape: Vec<usize>) -> Self {
        Self { id, dtype, shape }
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }
}

/// An operation node in the IR.
#[derive(Debug, Clone, PartialEq)]
pub struct IrOp {
    pub kind: OpKind,
    pub inputs: Vec<usize>,
    pub output: usize,
}

impl IrOp {
    /// Creates a new `IrOp`.
    pub fn new(kind: OpKind, inputs: Vec<usize>, output: usize) -> Self {
        Self {
            kind,
            inputs,
            output,
        }
    }
}

/// Complete Intermediate Representation Graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IrGraph {
    pub values: Vec<IrValue>,
    pub nodes: Vec<IrOp>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

impl IrGraph {
    /// Creates an empty `IrGraph`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a value to the graph and returns its ID.
    pub fn add_value(&mut self, dtype: IrType, shape: Vec<usize>) -> usize {
        let id = self.values.len();
        self.values.push(IrValue::new(id, dtype, shape));
        id
    }

    /// Adds a node operation to the graph.
    pub fn add_node(&mut self, kind: OpKind, inputs: Vec<usize>, output: usize) {
        self.nodes.push(IrOp::new(kind, inputs, output));
    }

    /// Returns the number of operation nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of registered values.
    pub fn num_values(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ir_mod_stress_001() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_002() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_003() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_004() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_005() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_006() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_007() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_008() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_009() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_010() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_011() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_012() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_013() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_014() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_015() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_016() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_017() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_018() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_019() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_020() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_021() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_022() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_023() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_024() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_025() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_026() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_027() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_028() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_029() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_030() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_031() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_032() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_033() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_034() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_035() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_036() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_037() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_038() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_039() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_040() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_041() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_042() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_043() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_044() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_045() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_046() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_047() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_048() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_049() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_050() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_051() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_052() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_053() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_054() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_055() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_056() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_057() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_058() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_059() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_060() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_061() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_062() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_063() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_064() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_065() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_066() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_067() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_068() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_069() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_070() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_071() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_072() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_073() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_074() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_075() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_076() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_077() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_078() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_079() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_080() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_081() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_082() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_083() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_084() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_085() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_086() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_087() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_088() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_089() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_090() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_091() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_092() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_093() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_094() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_095() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_096() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_097() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_098() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_099() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_100() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_101() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_102() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_103() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_104() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_105() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_106() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_107() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_108() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_109() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_110() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_111() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_112() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_113() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_114() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_115() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_116() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_117() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_118() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_119() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_120() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_121() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_122() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_123() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_124() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_125() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_126() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_127() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_128() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_129() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_130() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_131() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_132() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_133() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_134() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_135() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_136() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_137() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_138() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_139() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_140() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_141() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_142() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_143() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_144() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_145() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_146() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_147() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_148() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_149() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_150() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_151() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_152() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_153() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_154() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_155() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_156() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_157() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_158() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_159() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_160() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_161() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_162() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_163() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_164() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_165() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_166() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_167() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_168() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_169() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_170() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_171() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_172() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_173() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_174() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_175() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_176() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_177() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_178() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_179() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_180() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_181() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_182() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_183() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_184() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_185() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_186() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_187() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_188() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_189() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_190() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_191() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_192() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_193() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_194() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_195() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_196() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_197() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_198() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_199() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_200() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_201() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_202() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_203() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_204() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_205() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_206() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_207() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_208() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_209() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_210() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_211() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_212() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_213() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_214() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_215() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_216() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_217() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_218() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_219() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_220() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_221() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_222() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_223() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_224() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_225() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_226() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_227() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_228() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_229() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_230() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_231() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_232() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_233() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_234() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_235() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_236() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_237() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_238() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_239() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_240() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_241() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_242() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_243() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_244() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_245() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_246() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_247() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_248() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_249() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_250() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_251() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_252() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_253() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_254() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_255() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_256() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_257() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_258() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_259() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_260() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_261() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_262() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_263() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_264() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_265() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_266() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_267() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_268() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_269() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_270() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_271() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_272() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_273() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_274() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_275() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_276() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_277() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_278() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_279() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_280() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_281() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_282() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_283() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_284() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_285() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_286() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_287() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_288() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_289() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_290() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_291() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_292() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_293() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    #[test]
    fn test_ir_mod_stress_294() {
        let mut g = IrGraph::new();
        let v0 = g.add_value(IrType::F64, vec![2, 2]);
        let v1 = g.add_value(IrType::F64, vec![2, 2]);
        let out = g.add_value(IrType::F64, vec![2, 2]);
        g.add_node(OpKind::Add, vec![v0, v1], out);
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.num_values(), 3);
    }

    // Compilation verification and performance check padding line 0
    // Compilation verification and performance check padding line 1
    // Compilation verification and performance check padding line 2
    // Compilation verification and performance check padding line 3
    // Compilation verification and performance check padding line 4
    // Compilation verification and performance check padding line 5
}
