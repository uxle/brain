//! # Graph Convolutional Network (GCN) Layer
//!
//! GCN convolution layer: H^(l+1) = sigma( D^-1/2 A D^-1/2 H^(l) W^(l) ).
#![allow(missing_docs)]

use brain_core::Tensor;
use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::aggregate_mean;
use crate::impl_::transform_node_features;

/// GCN Layer struct.
#[derive(Debug, Clone)]
pub struct GcnLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl GcnLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weight = Tensor::zeros(vec![out_dim, in_dim]);
        let bias = Some(Tensor::zeros(vec![out_dim]));
        Self { in_dim, out_dim, weight, bias }
    }
}

impl GnnLayer for GcnLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        // Transform node features by weight
        let h_transformed = transform_node_features(x, &self.weight, self.bias.as_ref());
        // Message passing: aggregate neighborhood
        let aggregated = aggregate_mean(&h_transformed, &graph.dst_nodes, graph.num_nodes);
        // ReLU activation
        let data: Vec<f64> = aggregated.to_vec().iter().map(|&v| v.max(0.0)).collect();
        Tensor::from_vec(data, aggregated.shape().to_vec())
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
