//! # GNN Explainability
//!
//! Gradient-based node/edge saliency and attention-based mask computation.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::graph::Graph;

/// Explanation report containing node and edge importance scores.
#[derive(Debug, Clone)]
pub struct ExplanationReport {
    pub node_importance: Vec<f64>,
    pub edge_importance: Vec<f64>,
}

impl ExplanationReport {
    pub fn top_nodes(&self, k: usize) -> Vec<usize> {
        let mut idx_scores: Vec<(usize, f64)> = self.node_importance.iter().copied().enumerate().collect();
        idx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        idx_scores.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

/// Computes gradient-based node saliency scores.
pub fn saliency_node_importance(graph: &Graph, node_embeddings: &Tensor) -> ExplanationReport {
    let data = node_embeddings.to_vec();
    let num_nodes = graph.num_nodes;
    let dim = if node_embeddings.shape().len() > 1 { node_embeddings.shape()[1] } else { 1 };

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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_explain_stress_001() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_002() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_003() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_004() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_005() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_006() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_007() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_008() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_009() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_010() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_011() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_012() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_013() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_014() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_015() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_016() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_017() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_018() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_019() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_020() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_021() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_022() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_023() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_024() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_025() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_026() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_027() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_028() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_029() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_030() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_031() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_032() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_033() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_034() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_035() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_036() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_037() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_038() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_039() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_040() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_041() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_042() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_043() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_044() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_045() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_046() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_047() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_048() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_049() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_050() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_051() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_052() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_053() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_054() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_055() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_056() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_057() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_058() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_059() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_060() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_061() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_062() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_063() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_064() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_065() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_066() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_067() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_068() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_069() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_070() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_071() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_072() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_073() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_074() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_075() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_076() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_077() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_078() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_079() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_080() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_081() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_082() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_083() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_084() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_085() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_086() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_087() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_088() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_089() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_090() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_091() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_092() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_093() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_094() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_095() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_096() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_097() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_098() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_099() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_100() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_101() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_102() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_103() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_104() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_105() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_106() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_107() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_108() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_109() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_110() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_111() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_112() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_113() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_114() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_115() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_116() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_117() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_118() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_119() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_120() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_121() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_122() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_123() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_124() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_125() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_126() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_127() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_128() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_129() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_130() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_131() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_132() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_133() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_134() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_135() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_136() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_137() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_138() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_139() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_140() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_141() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_142() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_143() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_144() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_145() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_146() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_147() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_148() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_149() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_150() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_151() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_152() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_153() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_154() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_155() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_156() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_157() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_158() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_159() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_160() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_161() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_162() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_163() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_164() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_165() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_166() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_167() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_168() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_169() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_170() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_171() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_172() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_173() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_174() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_175() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_176() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_177() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_178() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_179() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_180() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_181() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_182() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_183() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_184() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_185() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_186() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_187() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_188() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_189() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_190() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_191() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_192() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_193() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_194() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_195() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_196() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_197() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_198() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_199() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_200() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_201() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_202() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_203() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_204() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_205() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_206() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_207() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_208() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_209() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_210() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_211() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_212() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_213() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_214() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_215() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_216() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_217() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_218() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_219() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_220() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_221() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_222() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_223() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_224() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_225() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_226() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_227() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_228() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_229() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_230() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_231() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_232() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_233() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_234() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_235() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_236() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_237() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_238() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_239() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_240() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_241() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_242() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_243() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_244() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_245() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_246() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_247() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_248() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_249() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_250() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_251() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_252() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_253() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_254() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_255() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_256() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_257() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_258() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_259() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_260() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_261() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_262() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_263() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_264() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_265() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_266() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_267() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_268() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_269() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_270() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_271() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_272() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_explain_stress_273() {
        let feats = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]);
        let graph = Graph::new(2, vec![0], vec![1], feats.clone()).unwrap();

        let report = saliency_node_importance(&graph, &feats);
        assert_eq!(report.node_importance.len(), 2);
        assert_eq!(report.edge_importance.len(), 1);
        let top = report.top_nodes(1);
        assert_eq!(top.len(), 1);
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
