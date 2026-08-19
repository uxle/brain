//! # GNN Core Types
//!
//! Fundamental data structures: NodeIndex, EdgeIndex, GraphTensor, BatchGraph.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Node identifier type.
pub type NodeIndex = usize;

/// Edge identifier type.
pub type EdgeIndex = usize;

/// Error type for GNN operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GnnError {
    InvalidGraph(String),
    NodeOutOfBounds { index: usize, max: usize },
    EdgeOutOfBounds { index: usize, max: usize },
    DimensionMismatch { expected: usize, got: usize },
    TrainingFailed(String),
}

impl std::fmt::Display for GnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GnnError::InvalidGraph(s) => write!(f, "Invalid graph: {}", s),
            GnnError::NodeOutOfBounds { index, max } => write!(f, "Node index {} out of bounds (max {})", index, max),
            GnnError::EdgeOutOfBounds { index, max } => write!(f, "Edge index {} out of bounds (max {})", index, max),
            GnnError::DimensionMismatch { expected, got } => write!(f, "Dim mismatch: expected {}, got {}", expected, got),
            GnnError::TrainingFailed(s) => write!(f, "Training failed: {}", s),
        }
    }
}

pub type GnnResult<T> = Result<T, GnnError>;

/// Tensor representation of graph adjacency matrix.
#[derive(Debug, Clone)]
pub struct GraphTensor {
    pub adj: Tensor,
    pub node_features: Tensor,
    pub edge_features: Option<Tensor>,
}

impl GraphTensor {
    pub fn new(adj: Tensor, node_features: Tensor) -> Self {
        Self { adj, node_features, edge_features: None }
    }

    pub fn num_nodes(&self) -> usize {
        self.node_features.shape()[0]
    }

    pub fn feature_dim(&self) -> usize {
        if self.node_features.shape().len() > 1 {
            self.node_features.shape()[1]
        } else {
            1
        }
    }
}

/// Disjoint union of multiple graphs into a batch.
#[derive(Debug, Clone)]
pub struct BatchGraph {
    pub src_nodes: Vec<NodeIndex>,
    pub dst_nodes: Vec<NodeIndex>,
    pub node_features: Tensor,
    pub batch_offsets: Vec<usize>,
    pub graph_ids: Vec<usize>,
}

impl BatchGraph {
    pub fn new(
        src_nodes: Vec<NodeIndex>,
        dst_nodes: Vec<NodeIndex>,
        node_features: Tensor,
        batch_offsets: Vec<usize>,
        graph_ids: Vec<usize>,
    ) -> Self {
        Self { src_nodes, dst_nodes, node_features, batch_offsets, graph_ids }
    }

    pub fn num_graphs(&self) -> usize {
        self.batch_offsets.len()
    }

    pub fn total_nodes(&self) -> usize {
        self.node_features.shape()[0]
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
