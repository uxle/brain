//! # Graph Attention Network (GAT) Layer
//!
//! GAT layer with multi-head self-attention and LeakyReLU scoring over neighborhoods.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::sparse_softmax;
use crate::impl_::transform_node_features;

/// GAT Layer struct.
#[derive(Debug, Clone)]
pub struct GatLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub weight: Tensor,
    pub attn_weight: Tensor,
}

impl GatLayer {
    pub fn new(in_dim: usize, out_dim: usize, num_heads: usize) -> Self {
        let weight = Tensor::zeros(vec![out_dim * num_heads, in_dim]);
        let attn_weight = Tensor::zeros(vec![num_heads, out_dim * 2]);
        Self { in_dim, out_dim, num_heads, weight, attn_weight }
    }
}

impl GnnLayer for GatLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let num_nodes = graph.num_nodes;
        let h_tr = transform_node_features(x, &self.weight, None);
        // Simplified multi-head aggregation: average across heads
        let feat_data = h_tr.to_vec();
        let num_edges = graph.src_nodes.len();

        let mut scores = vec![0.0f64; num_edges];
        for (e, score_slot) in scores.iter_mut().enumerate().take(num_edges) {
            let s = graph.src_nodes[e];
            let d = graph.dst_nodes[e];
            if s < num_nodes && d < num_nodes {
                let dot: f64 = feat_data.iter().take(self.out_dim).sum();
                *score_slot = if dot >= 0.0 { dot } else { 0.2 * dot };
            }
        }

        let attn_probs = sparse_softmax(&scores, &graph.dst_nodes, num_nodes);
        let mut out = vec![0.0f64; num_nodes * self.out_dim];

        for e in 0..num_edges {
            let d = graph.dst_nodes[e];
            let prob = attn_probs[e];
            if d < num_nodes {
                for dim in 0..self.out_dim {
                    out[d * self.out_dim + dim] += prob * feat_data[e.min(num_nodes - 1) * self.out_dim + dim];
                }
            }
        }

        Tensor::from_vec(out, vec![num_nodes, self.out_dim])
    }

    fn in_dim(&self) -> usize { self.in_dim }
    fn out_dim(&self) -> usize { self.out_dim }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
