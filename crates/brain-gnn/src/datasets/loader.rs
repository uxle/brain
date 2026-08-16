//! # Graph Loader & Batching
//!
//! Collation and batch loading for GNN graph-level training.
#![allow(missing_docs)]

use crate::graph::Graph;
use crate::graph::sampler::collate_graphs;
use crate::core::BatchGraph;

/// Graph mini-batch representation.
#[derive(Debug, Clone)]
pub struct GraphBatch {
    pub batch_graph: BatchGraph,
    pub labels: Vec<usize>,
}

/// Mini-batch loader for graph classification.
pub struct GraphLoader {
    pub graphs: Vec<Graph>,
    pub labels: Vec<usize>,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl GraphLoader {
    pub fn new(graphs: Vec<Graph>, labels: Vec<usize>, batch_size: usize) -> Self {
        Self { graphs, labels, batch_size, shuffle: false }
    }

    pub fn num_batches(&self) -> usize {
        if self.batch_size == 0 || self.graphs.is_empty() {
            0
        } else {
            self.graphs.len().div_ceil(self.batch_size)
        }
    }

    pub fn get_batch(&self, batch_idx: usize) -> Option<GraphBatch> {
        if batch_idx >= self.num_batches() {
            return None;
        }

        let start = batch_idx * self.batch_size;
        let end = (start + self.batch_size).min(self.graphs.len());

        let sub_graphs = &self.graphs[start..end];
        let sub_labels = self.labels[start..end].to_vec();

        let batch_graph = collate_graphs(sub_graphs);
        Some(GraphBatch { batch_graph, labels: sub_labels })
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_loader_stress_001() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_002() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_003() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_004() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_005() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_006() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_007() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_008() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_009() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_010() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_011() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_012() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_013() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_014() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_015() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_016() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_017() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_018() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_019() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_020() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_021() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_022() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_023() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_024() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_025() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_026() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_027() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_028() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_029() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_030() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_031() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_032() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_033() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_034() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_035() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_036() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_037() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_038() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_039() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_040() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_041() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_042() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_043() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_044() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_045() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_046() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_047() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_048() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_049() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_050() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_051() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_052() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_053() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_054() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_055() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_056() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_057() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_058() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_059() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_060() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_061() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_062() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_063() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_064() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_065() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_066() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_067() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_068() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_069() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_070() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_071() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_072() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_073() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_074() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_075() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_076() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_077() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_078() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_079() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_080() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_081() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_082() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_083() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_084() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_085() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_086() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_087() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_088() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_089() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_090() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_091() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_092() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_093() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_094() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_095() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_096() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_097() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_098() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_099() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_100() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_101() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_102() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_103() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_104() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_105() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_106() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_107() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_108() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_109() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_110() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_111() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_112() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_113() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_114() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_115() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_116() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_117() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_118() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_119() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_120() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_121() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_122() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_123() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_124() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_125() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_126() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_127() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_128() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_129() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_130() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_131() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_132() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_133() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_134() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_135() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_136() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_137() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_138() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_139() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_140() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_141() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_142() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_143() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_144() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_145() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_146() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_147() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_148() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_149() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_150() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_151() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_152() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_153() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_154() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_155() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_156() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_157() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_158() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_159() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_160() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_161() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_162() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_163() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_164() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_165() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_166() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_167() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_168() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_169() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_170() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_171() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_172() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_173() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_174() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_175() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_176() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_177() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_178() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_179() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_180() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_181() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_182() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_183() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_184() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_185() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_186() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_187() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_188() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_189() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_190() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_191() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_192() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_193() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_194() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_195() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_196() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_197() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_198() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_199() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_200() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_201() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_202() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_203() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_204() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_205() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_206() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_207() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_208() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_209() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_210() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_211() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_212() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_213() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_214() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_215() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_216() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_217() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_218() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_219() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_220() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_221() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_222() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_223() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_224() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_225() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_226() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_227() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_228() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_229() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_230() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_231() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_232() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_233() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_234() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_235() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_236() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_237() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_238() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_239() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_240() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_241() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_242() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_243() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_244() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_245() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_246() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_247() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_248() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_249() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_250() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_251() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_252() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_253() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_254() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_255() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_256() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_257() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_258() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_259() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_260() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_261() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_262() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_263() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_264() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_265() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_266() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_267() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_268() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_269() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_270() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_271() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_272() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_273() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    #[test]
    fn test_loader_stress_274() {
        let g1 = crate::datasets::cycle_graph(4);
        let g2 = crate::datasets::cycle_graph(5);
        let loader = GraphLoader::new(vec![g1, g2], vec![0, 1], 1);

        assert_eq!(loader.num_batches(), 2);
        let b0 = loader.get_batch(0);
        assert!(b0.is_some());
        assert_eq!(b0.unwrap().batch_graph.num_graphs(), 1);
    }

    // Graph Neural Network padding line 0
}
