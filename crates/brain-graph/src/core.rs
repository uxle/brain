//! # Graph Core Types
//!
//! Fundamental IDs, data types, device kinds, shapes, and error models.
#![allow(missing_docs)]


/// Unique identifier for a graph node.
pub type NodeId = usize;

/// Unique identifier for an intermediate tensor value.
pub type ValueId = usize;

/// Unique identifier for an edge in the computation graph.
pub type EdgeId = usize;

/// Supported data types in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    #[default]
    F32,
    F64,
    I32,
    I64,
    Bool,
}

/// Target execution device category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceKind {
    #[default]
    Cpu,
    Cuda(usize),
    Wasm,
}

/// Tensor shape descriptor supporting symbolic dynamic dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() { 0 } else { self.dims.iter().product() }
    }
}

/// Error type for computation graph operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    NodeNotFound(NodeId),
    ValueNotFound(ValueId),
    CyclicDependency(String),
    TypeMismatch { expected: DType, got: DType },
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    VerificationFailed(String),
    PassFailed(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeNotFound(id) => write!(f, "Node {} not found in graph", id),
            GraphError::ValueNotFound(id) => write!(f, "Value {} not found in graph", id),
            GraphError::CyclicDependency(msg) => write!(f, "Cycle detected: {}", msg),
            GraphError::TypeMismatch { expected, got } => write!(f, "Type mismatch: expected {:?}, got {:?}", expected, got),
            GraphError::ShapeMismatch { expected, got } => write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got),
            GraphError::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            GraphError::PassFailed(msg) => write!(f, "Pass failed: {}", msg),
        }
    }
}

pub type GraphResult<T> = Result<T, GraphError>;

/// Metadata associated with an entire computation graph.
#[derive(Debug, Clone, Default)]
pub struct GraphMetadata {
    pub name: String,
    pub version: usize,
    pub author: String,
    pub target_device: DeviceKind,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
}
