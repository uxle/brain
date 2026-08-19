//! # Graph Data Structure
//!
//! CSR / Edge List graph representation with node and edge features.
#![allow(missing_docs)]

pub mod ops;
pub mod sampler;

pub use ops::*;
pub use sampler::*;

use brain_core::Tensor;
use crate::core::GnnError;

/// Configuration parameters for graph creation.
#[derive(Debug, Clone, Default)]
pub struct GraphConfig {
    pub is_directed: bool,
    pub allow_self_loops: bool,
    pub allow_multi_edges: bool,
}

/// Core Graph structure representing node/edge features and adjacency.
#[derive(Debug, Clone)]
pub struct Graph {
    pub num_nodes: usize,
    pub src_nodes: Vec<usize>,
    pub dst_nodes: Vec<usize>,
    pub edge_weights: Option<Vec<f64>>,
    pub node_features: Tensor,
    pub edge_features: Option<Tensor>,
    pub config: GraphConfig,
}

impl Graph {
    pub fn new(
        num_nodes: usize,
        src_nodes: Vec<usize>,
        dst_nodes: Vec<usize>,
        node_features: Tensor,
    ) -> Result<Self, GnnError> {
        if src_nodes.len() != dst_nodes.len() {
            return Err(GnnError::InvalidGraph("src and dst length mismatch".into()));
        }
        for &s in &src_nodes {
            if s >= num_nodes {
                return Err(GnnError::NodeOutOfBounds { index: s, max: num_nodes });
            }
        }
        for &d in &dst_nodes {
            if d >= num_nodes {
                return Err(GnnError::NodeOutOfBounds { index: d, max: num_nodes });
            }
        }
        if node_features.shape()[0] != num_nodes {
            return Err(GnnError::DimensionMismatch {
                expected: num_nodes,
                got: node_features.shape()[0],
            });
        }
        Ok(Self {
            num_nodes,
            src_nodes,
            dst_nodes,
            edge_weights: None,
            node_features,
            edge_features: None,
            config: GraphConfig::default(),
        })
    }

    pub fn num_edges(&self) -> usize {
        self.src_nodes.len()
    }

    pub fn feature_dim(&self) -> usize {
        if self.node_features.shape().len() > 1 {
            self.node_features.shape()[1]
        } else {
            1
        }
    }

    pub fn degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.num_nodes];
        for &s in &self.src_nodes {
            if s < self.num_nodes {
                deg[s] += 1;
            }
        }
        deg
    }

    pub fn validate(&self) -> Result<(), GnnError> {
        if self.node_features.shape()[0] != self.num_nodes {
            return Err(GnnError::InvalidGraph("Node feature count mismatch".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
