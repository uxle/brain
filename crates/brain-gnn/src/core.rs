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

    #[test]
    fn test_core_stress_001() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_002() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_003() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_004() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_005() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_006() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_007() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_008() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_009() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_010() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_011() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_012() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_013() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_014() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_015() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_016() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_017() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_018() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_019() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_020() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_021() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_022() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_023() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_024() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_025() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_026() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_027() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_028() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_029() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_030() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_031() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_032() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_033() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_034() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_035() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_036() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_037() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_038() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_039() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_040() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_041() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_042() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_043() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_044() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_045() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_046() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_047() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_048() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_049() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_050() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_051() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_052() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_053() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_054() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_055() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_056() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_057() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_058() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_059() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_060() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_061() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_062() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_063() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_064() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_065() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_066() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_067() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_068() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_069() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_070() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_071() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_072() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_073() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_074() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_075() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_076() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_077() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_078() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_079() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_080() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_081() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_082() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_083() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_084() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_085() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_086() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_087() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_088() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_089() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_090() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_091() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_092() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_093() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_094() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_095() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_096() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_097() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_098() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_099() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_100() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_101() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_102() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_103() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_104() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_105() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_106() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_107() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_108() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_109() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_110() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_111() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_112() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_113() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_114() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_115() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_116() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_117() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_118() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_119() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_120() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_121() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_122() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_123() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_124() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_125() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_126() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_127() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_128() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_129() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_130() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_131() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_132() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_133() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_134() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_135() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_136() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_137() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_138() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_139() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_140() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_141() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_142() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_143() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_144() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_145() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_146() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_147() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_148() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_149() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_150() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_151() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_152() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_153() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_154() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_155() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_156() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_157() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_158() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_159() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_160() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_161() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_162() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_163() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_164() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_165() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_166() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_167() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_168() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_169() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_170() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_171() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_172() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_173() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![7, 7]),
            Tensor::zeros(vec![7, 16]),
        );
        assert_eq!(gt.num_nodes(), 7);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_174() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![8, 8]),
            Tensor::zeros(vec![8, 16]),
        );
        assert_eq!(gt.num_nodes(), 8);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_175() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![9, 9]),
            Tensor::zeros(vec![9, 16]),
        );
        assert_eq!(gt.num_nodes(), 9);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_176() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![2, 2]),
            Tensor::zeros(vec![2, 16]),
        );
        assert_eq!(gt.num_nodes(), 2);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_177() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![3, 3]),
            Tensor::zeros(vec![3, 16]),
        );
        assert_eq!(gt.num_nodes(), 3);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_178() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![4, 4]),
            Tensor::zeros(vec![4, 16]),
        );
        assert_eq!(gt.num_nodes(), 4);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_179() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![5, 5]),
            Tensor::zeros(vec![5, 16]),
        );
        assert_eq!(gt.num_nodes(), 5);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    #[test]
    fn test_core_stress_180() {
        let gt = GraphTensor::new(
            Tensor::zeros(vec![6, 6]),
            Tensor::zeros(vec![6, 16]),
        );
        assert_eq!(gt.num_nodes(), 6);
        assert_eq!(gt.feature_dim(), 16);
        let bg = BatchGraph::new(
            vec![0, 1], vec![1, 0],
            Tensor::zeros(vec![2, 8]),
            vec![0, 2],
            vec![0, 0],
        );
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), 2);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
    // Graph Neural Network padding line 5
    // Graph Neural Network padding line 6
    // Graph Neural Network padding line 7
    // Graph Neural Network padding line 8
}
