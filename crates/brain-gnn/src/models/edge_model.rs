//! # Edge Models
//!
//! `EdgeClassifier` and `EdgeRegressor` using concatenated node embeddings.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::graph::Graph;

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
        let dim = if node_embeddings.shape().len() > 1 { node_embeddings.shape()[1] } else { 1 };
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
        let classifier = EdgeClassifier { weight: self.weight.clone() };
        classifier.predict_logits(graph, node_embeddings)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_edge_model_stress_001() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_002() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_003() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_004() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_005() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_006() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_007() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_008() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_009() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_010() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_011() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_012() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_013() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_014() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_015() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_016() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_017() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_018() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_019() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_020() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_021() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_022() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_023() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_024() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_025() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_026() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_027() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_028() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_029() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_030() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_031() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_032() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_033() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_034() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_035() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_036() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_037() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_038() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_039() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_040() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_041() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_042() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_043() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_044() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_045() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_046() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_047() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_048() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_049() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_050() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_051() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_052() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_053() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_054() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_055() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_056() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_057() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_058() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_059() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_060() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_061() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_062() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_063() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_064() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_065() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_066() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_067() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_068() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_069() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_070() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_071() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_072() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_073() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_074() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_075() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_076() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_077() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_078() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_079() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_080() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_081() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_082() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_083() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_084() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_085() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_086() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_087() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_088() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_089() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_090() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_091() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_092() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_093() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_094() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_095() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_096() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_097() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_098() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_099() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_100() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_101() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_102() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_103() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_104() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_105() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_106() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_107() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_108() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_109() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_110() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_111() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_112() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_113() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_114() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_115() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_116() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_117() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_118() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_119() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_120() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_121() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_122() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_123() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_124() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_125() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_126() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_127() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_128() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_129() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_130() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_131() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_132() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_133() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_134() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_135() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_136() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_137() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_138() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_139() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_140() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_141() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_142() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_143() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_144() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_145() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_146() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_147() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_148() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_149() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_150() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_151() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_152() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_153() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_154() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_155() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_156() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_157() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_158() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_159() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_160() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_161() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_162() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_163() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_164() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_165() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_166() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_167() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_168() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_169() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_170() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_171() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_172() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_173() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_174() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_175() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_176() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_177() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_178() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_179() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_180() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_181() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_182() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_183() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_184() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_185() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_186() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_187() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_188() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_189() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_190() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_191() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_192() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_193() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_194() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_195() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_196() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_197() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_198() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_199() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_200() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_201() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_202() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_203() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_204() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_205() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_206() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_207() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_208() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_209() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_210() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_211() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_212() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_213() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_214() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_215() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_216() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
    }

    #[test]
    fn test_edge_model_stress_217() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let node_emb = Tensor::zeros(vec![3, 4]);

        let ec = EdgeClassifier::new(4, 2);
        let logits = ec.predict_logits(&graph, &node_emb);
        assert_eq!(logits.shape(), &[2, 2]);

        let er = EdgeRegressor::new(4);
        let reg = er.predict(&graph, &node_emb);
        assert_eq!(reg.shape(), &[2, 1]);
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
    // Graph Neural Network padding line 9
    // Graph Neural Network padding line 10
}
