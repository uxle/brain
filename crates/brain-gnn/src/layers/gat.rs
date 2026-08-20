//! # Graph Attention Network (GAT) Layer (Veličković et al.)
//!
//! Multi-head self-attention with LeakyReLU scoring over graph neighborhoods:
//! \alpha_{ij}^k = \text{Softmax}_j( \text{LeakyReLU}( a_k^T [W_k h_i \,\|\, W_k h_j] ) )
#![allow(missing_docs)]

use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::sparse_softmax;
use brain_core::Tensor;

/// GAT Layer struct implementing Multi-Head Graph Attention.
#[derive(Debug, Clone)]
pub struct GatLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub weight: Tensor,
    pub attn_weight: Tensor,
    pub bias: Option<Tensor>,
}

impl GatLayer {
    /// Creates a new `GatLayer`.
    pub fn new(in_dim: usize, out_dim: usize, num_heads: usize) -> Self {
        let weight = Tensor::zeros(vec![num_heads, in_dim, out_dim]);
        let attn_weight = Tensor::zeros(vec![num_heads, out_dim * 2]);
        let bias = Some(Tensor::zeros(vec![out_dim]));
        Self {
            in_dim,
            out_dim,
            num_heads: num_heads.max(1),
            weight,
            attn_weight,
            bias,
        }
    }

    /// Forward pass computing multi-head graph attention.
    pub fn forward_gat(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let num_nodes = graph.num_nodes;
        assert_eq!(x.shape()[0], num_nodes, "Nodes mismatch");
        let x_data = x.data();

        let num_edges = graph.src_nodes.len();
        let mut head_outputs = vec![0.0f64; num_nodes * self.out_dim];

        for h in 0..self.num_heads {
            // 1. Linear projection: Wh_i [num_nodes, out_dim]
            let mut wh = vec![0.0f64; num_nodes * self.out_dim];
            let w_offset = h * self.in_dim * self.out_dim;
            let w_slice = &self.weight.data()[w_offset..w_offset + self.in_dim * self.out_dim];

            for i in 0..num_nodes {
                for j in 0..self.out_dim {
                    let mut sum = 0.0f64;
                    for k in 0..self.in_dim {
                        sum += x_data[i * self.in_dim + k] * w_slice[k * self.out_dim + j];
                    }
                    wh[i * self.out_dim + j] = sum;
                }
            }

            // 2. Attention scoring over edges e = (s, d)
            let a_offset = h * self.out_dim * 2;
            let a_slice = &self.attn_weight.data()[a_offset..a_offset + self.out_dim * 2];

            let mut edge_scores = vec![0.0f64; num_edges];
            for e in 0..num_edges {
                let s = graph.src_nodes[e];
                let d = graph.dst_nodes[e];
                if s < num_nodes && d < num_nodes {
                    let mut e_score = 0.0f64;
                    for dim in 0..self.out_dim {
                        e_score += a_slice[dim] * wh[s * self.out_dim + dim];
                        e_score += a_slice[self.out_dim + dim] * wh[d * self.out_dim + dim];
                    }
                    // LeakyReLU with slope 0.2
                    edge_scores[e] = if e_score >= 0.0 {
                        e_score
                    } else {
                        0.2 * e_score
                    };
                }
            }

            // 3. Sparse Softmax normalization over incoming edges per node
            let alpha = sparse_softmax(&edge_scores, &graph.dst_nodes, num_nodes);

            // 4. Neighborhood aggregation
            for e in 0..num_edges {
                let s = graph.src_nodes[e];
                let d = graph.dst_nodes[e];
                if s < num_nodes && d < num_nodes {
                    let prob = if e < alpha.len() { alpha[e] } else { 0.0 };
                    for dim in 0..self.out_dim {
                        head_outputs[d * self.out_dim + dim] += prob * wh[s * self.out_dim + dim];
                    }
                }
            }
        }

        // 5. Average across attention heads & add bias
        let head_scale = 1.0 / (self.num_heads as f64);
        for i in 0..num_nodes {
            for dim in 0..self.out_dim {
                let idx = i * self.out_dim + dim;
                head_outputs[idx] *= head_scale;
                if let Some(ref b) = self.bias {
                    head_outputs[idx] += b.data()[dim];
                }
                // ELU / ReLU activation
                head_outputs[idx] = head_outputs[idx].max(0.0);
            }
        }

        Tensor::from_vec(head_outputs, vec![num_nodes, self.out_dim])
    }
}

impl GnnLayer for GatLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        self.forward_gat(graph, x)
    }

    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.out_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gat_layer_multihead() {
        let src = vec![0, 1, 0];
        let dst = vec![1, 2, 2];
        let feats = Tensor::from_slice(&[1.0; 12], vec![3, 4]);
        let graph = Graph::new(3, src, dst, feats.clone()).unwrap();

        let gat = GatLayer::new(4, 2, 2);
        let h = gat.forward(&graph, &feats);

        assert_eq!(h.shape(), &[3, 2]);
    }
}
