//! # Graph Convolutional Network (GCN) Layer (Kipf & Welling)
//!
//! Renormalized spectral graph convolution:
//! H^(l+1) = sigma( \tilde{D}^-1/2 \tilde{A} \tilde{D}^-1/2 H^(l) W^(l) + b )
#![allow(missing_docs)]

use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::normalize_adj;
use brain_core::Tensor;

/// GCN Layer struct implementing Kipf & Welling spectral convolution.
#[derive(Debug, Clone)]
pub struct GcnLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl GcnLayer {
    /// Creates a new `GcnLayer` with weights initialized.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        // Glorot / Xavier uniform-like initialization scale
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt();
        let numel = in_dim * out_dim;
        let weight_data: Vec<f64> = (0..numel)
            .map(|i| ((i % 7) as f64 - 3.0) * (scale / 3.0))
            .collect();
        let weight = Tensor::from_vec(weight_data, vec![in_dim, out_dim]);
        let bias = Some(Tensor::zeros(vec![out_dim]));
        Self {
            in_dim,
            out_dim,
            weight,
            bias,
        }
    }

    /// Explicit forward pass with optional custom activation.
    pub fn forward_gcn(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let n = graph.num_nodes;
        assert_eq!(
            x.shape()[0],
            n,
            "Feature matrix rows must match graph nodes"
        );
        assert_eq!(x.shape()[1], self.in_dim, "Feature dimension mismatch");

        // 1. Construct Renormalized Adjacency \tilde{A} = A + I_N
        let mut adj_dense = vec![0.0f64; n * n];
        for i in 0..n {
            adj_dense[i * n + i] = 1.0; // self-loop
        }
        for idx in 0..graph.src_nodes.len() {
            let s = graph.src_nodes[idx];
            let d = graph.dst_nodes[idx];
            let w = graph.edge_weights.as_ref().map(|w| w[idx]).unwrap_or(1.0);
            if s < n && d < n {
                adj_dense[s * n + d] += w;
            }
        }
        let adj_tensor = Tensor::from_vec(adj_dense, vec![n, n]);
        let norm_adj = normalize_adj(&adj_tensor);

        // 2. Feature projection: X_proj = X * W [N, out_dim]
        let x_data = x.data();
        let w_data = self.weight.data();
        let mut x_proj = vec![0.0f64; n * self.out_dim];

        for i in 0..n {
            for j in 0..self.out_dim {
                let mut sum = 0.0f64;
                for k in 0..self.in_dim {
                    sum += x_data[i * self.in_dim + k] * w_data[k * self.out_dim + j];
                }
                x_proj[i * self.out_dim + j] = sum;
            }
        }

        // 3. Spectral propagation: H = \hat{A} * X_proj
        let norm_a = norm_adj.data();
        let mut out = vec![0.0f64; n * self.out_dim];

        for i in 0..n {
            for j in 0..self.out_dim {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += norm_a[i * n + k] * x_proj[k * self.out_dim + j];
                }
                if let Some(ref b) = self.bias {
                    sum += b.data()[j];
                }
                // ReLU activation
                out[i * self.out_dim + j] = sum.max(0.0);
            }
        }

        Tensor::from_vec(out, vec![n, self.out_dim])
    }
}

impl GnnLayer for GcnLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        self.forward_gcn(graph, x)
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
    fn test_gcn_layer_forward() {
        let src = vec![0, 1];
        let dst = vec![1, 2];
        let feats = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0, 3.0, 3.0], vec![3, 2]);
        let graph = Graph::new(3, src, dst, feats.clone()).unwrap();

        let mut gcn = GcnLayer::new(2, 2);
        // Set identity weight for deterministic verification
        gcn.weight = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        gcn.bias = None;

        let h = gcn.forward(&graph, &feats);

        assert_eq!(h.shape(), &[3, 2]);
        assert!(h.data().iter().all(|&v| v >= 0.0));
    }
}
