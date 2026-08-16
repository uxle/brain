//! # brain-gnn
//!
//! Production-grade Graph Neural Network (GNN) framework in pure Rust:
//! CSR graphs, GCN, GAT, GraphSAGE, GIN, GatedConv, EdgeConv, GraphTransformer,
//! pooling/readouts, Jumping Knowledge, datasets, batch loading, and explainability.
//!
//! ## Architecture
//! - [`graph`] — `Graph` structure, CSR adjacency, degree ops, subgraphs, neighbor sampling
//! - [`layers`] — `GnnLayer` trait, GCN, GAT, GraphSAGE, GIN, GGCN, EdgeConv, GraphTransformer
//! - [`readout`] — global pool (add, mean, max), Jumping Knowledge aggregation
//! - [`models`] — GcnModel, GatModel, SageModel, GinModel, EdgeClassifier, EdgeRegressor
//! - [`train`] — `GnnTrainer`, `GnnTrainConfig`, metrics and statistics
//! - [`datasets`] — planted community graph, cycle graph, Zachary's Karate Club, GraphLoader
//! - [`explain`] — gradient/saliency node importance, ExplanationReport
//! - [`core`] — `GraphTensor`, `BatchGraph`, `NodeIndex`, `EdgeIndex`, `GnnError`
//! - [`config`] — `GnnConfig`, `LayerConfig`, `LayerType`, `AggregatorType`, `PoolingType`
//! - [`ops`] — symmetric adjacency normalization, sparse softmax, edge aggregation
//! - [`utils`] — k-NN graphs, radius graphs, self-loops, Erdős–Rényi random graphs
//! - [`builder`] — fluent `GnnBuilder` API

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod builder;
pub mod config;
pub mod core;
pub mod datasets;
pub mod explain;
pub mod graph;
pub mod impl_;
pub mod layers;
pub mod models;
pub mod ops;
pub mod readout;
pub mod train;
pub mod utils;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use builder::GnnBuilder;
pub use config::{
    AggregatorType, GnnConfig, LayerConfig, LayerType, PoolingType,
};
pub use core::{BatchGraph, EdgeIndex, GnnError, GnnResult, GraphTensor, NodeIndex};
pub use datasets::{
    DatasetSplits, GraphBatch, GraphLoader, cycle_graph, random_community_graph,
    zachary_karate_club,
};
pub use explain::{ExplanationReport, saliency_node_importance};
pub use graph::{
    Graph, GraphConfig, SampledSubgraph, in_degrees, induced_subgraph,
    normalized_graph_adj, out_degrees, sample_neighbors, to_dense_adj,
};
pub use impl_::{embed_nodes, transform_node_features};
pub use layers::{
    EdgeConv, GatLayer, GcnLayer, GinLayer, GatedConv, GnnLayer, GraphTransformerLayer,
    SageLayer,
};
pub use models::{
    EdgeClassifier, EdgeRegressor, GatModel, GcnModel, GinModel, SageModel,
};
pub use ops::{
    aggregate_max, aggregate_mean, aggregate_sum, normalize_adj, sparse_softmax,
};
pub use readout::{
    JkConfig, JkMode, JumpingKnowledge, global_add_pool, global_max_pool, global_mean_pool,
};
pub use train::{GnnTrainConfig, GnnTrainStats, GnnTrainer, TaskType};
pub use utils::{add_self_loops, knn_graph, radius_graph, random_graph_er};

/// Framework version string.
pub const VERSION: &str = "0.2.0";