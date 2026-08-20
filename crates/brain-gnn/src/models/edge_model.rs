//! # Edge Models
//!
//! `EdgeClassifier` and `EdgeRegressor` using concatenated node embeddings.
#![allow(missing_docs)]

use crate::graph::Graph;
use brain_core::Tensor;

/// Predicts discrete class labels for edges in a graph.
pub struct EdgeClassifier {
    pub weight: Tensor,
}

impl EdgeClassifier {
    pub fn new(node_dim: usize, num_classes: usize) -> Self {
        let weight = Tensor::zeros(vec![num_classes, node_dim * 2]);
        Self { weight }
    }

    pub fn predict_logits(&self, graph: &Graph, node_embeddings: &Tensor) -> Tensor {
        let num_edges = graph.src_nodes.len();
        let dim = if node_embeddings.shape().len() > 1 {
            node_embeddings.shape()[1]
        } else {
            1
        };
        let node_data = node_embeddings.to_vec();

        let mut edge_feats = vec![0.0f64; num_edges * dim * 2];
        for e in 0..num_edges {
            let s = graph.src_nodes[e];
            let d = graph.dst_nodes[e];
            if s < graph.num_nodes && d < graph.num_nodes {
                for i in 0..dim {
                    edge_feats[e * dim * 2 + i] = node_data[s * dim + i];
                    edge_feats[e * dim * 2 + dim + i] = node_data[d * dim + i];
                }
            }
        }

        let ef_tensor = Tensor::from_vec(edge_feats, vec![num_edges, dim * 2]);
        let out_classes = self.weight.shape()[0];

        // Linear projection
        let mut logits = vec![0.0f64; num_edges * out_classes];
        let w_data = self.weight.to_vec();
        let in_dim2 = dim * 2;
        let ef_data = ef_tensor.to_vec();

        for e in 0..num_edges {
            for c in 0..out_classes {
                let mut sum = 0.0f64;
                for i in 0..in_dim2 {
                    sum += ef_data[e * in_dim2 + i] * w_data[c * in_dim2 + i];
                }
                logits[e * out_classes + c] = sum;
            }
        }

        Tensor::from_vec(logits, vec![num_edges, out_classes])
    }
}

/// Predicts continuous values for edges in a graph.
pub struct EdgeRegressor {
    pub weight: Tensor,
}

impl EdgeRegressor {
    pub fn new(node_dim: usize) -> Self {
        let weight = Tensor::zeros(vec![1, node_dim * 2]);
        Self { weight }
    }

    pub fn predict(&self, graph: &Graph, node_embeddings: &Tensor) -> Tensor {
        let classifier = EdgeClassifier {
            weight: self.weight.clone(),
        };
        classifier.predict_logits(graph, node_embeddings)
    }
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
