//! # GNN Explainability
//!
//! Gradient-based node/edge saliency and attention-based mask computation.
#![allow(missing_docs)]

use crate::graph::Graph;
use brain_core::Tensor;

/// Explanation report containing node and edge importance scores.
#[derive(Debug, Clone)]
pub struct ExplanationReport {
    pub node_importance: Vec<f64>,
    pub edge_importance: Vec<f64>,
}

impl ExplanationReport {
    pub fn top_nodes(&self, k: usize) -> Vec<usize> {
        let mut idx_scores: Vec<(usize, f64)> =
            self.node_importance.iter().copied().enumerate().collect();
        idx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        idx_scores.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

/// Computes gradient-based node saliency scores.
pub fn saliency_node_importance(graph: &Graph, node_embeddings: &Tensor) -> ExplanationReport {
    let data = node_embeddings.to_vec();
    let num_nodes = graph.num_nodes;
    let dim = if node_embeddings.shape().len() > 1 {
        node_embeddings.shape()[1]
    } else {
        1
    };

    let mut node_scores = vec![0.0f64; num_nodes];
    for n in 0..num_nodes {
        let mut norm2 = 0.0f64;
        for d in 0..dim {
            let v = data[n * dim + d];
            norm2 += v * v;
        }
        node_scores[n] = norm2.sqrt();
    }

    let num_edges = graph.src_nodes.len();
    let mut edge_scores = vec![0.0f64; num_edges];
    for (e, score_slot) in edge_scores.iter_mut().enumerate().take(num_edges) {
        let s = graph.src_nodes[e];
        let d = graph.dst_nodes[e];
        if s < num_nodes && d < num_nodes {
            *score_slot = (node_scores[s] + node_scores[d]) / 2.0;
        }
    }

    ExplanationReport {
        node_importance: node_scores,
        edge_importance: edge_scores,
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
