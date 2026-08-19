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
}
