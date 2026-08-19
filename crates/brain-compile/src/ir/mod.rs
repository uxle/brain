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
}
