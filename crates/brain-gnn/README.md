# `brain-gnn`

Pure-Rust graph neural network framework: CSR graphs, GCN/GAT/SAGE/GIN/EdgeConv/Transformer layers, pooling, datasets, and explainability.

## Overview

`brain-gnn` implements graph deep learning on top of `brain-core` tensors with zero external dependencies. It provides a validated `Graph` structure with CSR adjacency, seven `GnnLayer` implementations, global pooling readouts with Jumping Knowledge, ready-made models and a `GnnTrainer`, synthetic/classic datasets, neighbor sampling, and gradient-saliency node explanation.

## Features

- **Graph structure**: `Graph::new` with validation, degree ops, dense/normalized adjacency, induced subgraphs, and `sample_neighbors` / `collate_graphs` for mini-batching.
- **Layers** (all behind the `GnnLayer` trait): `GcnLayer`, `GatLayer`, `SageLayer`, `GinLayer`, `GatedConv`, `EdgeConv`, `GraphTransformerLayer`.
- **Readouts**: `global_add_pool`, `global_mean_pool`, `global_max_pool`, and `JumpingKnowledge` (`JkConfig`/`JkMode`).
- **Models**: `GcnModel`, `GatModel`, `SageModel`, `GinModel` (node + graph tasks), `EdgeClassifier`, `EdgeRegressor`.
- **Training**: `GnnTrainer` with `GnnTrainConfig` (LR, weight decay, epochs, `TaskType`) and `GnnTrainStats` (loss/accuracy).
- **Datasets**: `GraphLoader`/`GraphBatch`, `DatasetSplits`, planted community graphs, cycle graphs, and Zachary's Karate Club.
- **Explainability**: `saliency_node_importance` producing an `ExplanationReport` with `top_nodes`.
- **Helpers**: `knn_graph`, `radius_graph`, `add_self_loops`, `random_graph_er`, and a fluent `GnnBuilder`.

## Modules

| Module | Contents |
|---|---|
| `graph` | `Graph`, `GraphConfig`, `SampledSubgraph`, degree/adjacency ops, neighbor sampling |
| `layers` | `GnnLayer` trait + 7 concrete layers |
| `readout` | global pooling + `JumpingKnowledge` |
| `models` | `GcnModel`, `GatModel`, `SageModel`, `GinModel`, edge models |
| `train` | `GnnTrainer`, `GnnTrainConfig`, `GnnTrainStats`, `TaskType` |
| `datasets` | `GraphLoader`, `GraphBatch`, `DatasetSplits`, synthetic generators |
| `explain` | `saliency_node_importance`, `ExplanationReport` |
| `core`/`config`/`ops`/`utils`/`builder` | shared types, configs, aggregation ops, graph utilities, `GnnBuilder` |

## Quick Start

```rust
use brain_core::Tensor;
use brain_gnn::{GcnModel, Graph};

let graph = Graph::new(
    4,
    vec![0, 1, 2, 3],
    vec![1, 2, 3, 0],
    Tensor::zeros(vec![4, 8]),
)
.unwrap();

let model = GcnModel::new(8, 16, 3, 2);
let node_out = model.forward_node(&graph);
```

## Testing

```bash
cargo test -p brain-gnn -j 2
```

## Workspace Role

Depends solely on `brain-core` for tensor storage and math, making `brain-gnn` a lightweight, self-contained graph-learning layer of the Brain framework.