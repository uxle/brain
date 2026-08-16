# `brain-gnn` (v0.2.0)

> Production-Grade Graph Neural Networks: CSR Adjacency, GCN, GAT, GraphSAGE, GIN, EdgeConv, Graph Transformers, Readouts, and Datasets.

## Overview

`brain-gnn` provides pure-Rust graph machine learning capabilities. It implements CSR graph storage, symmetric/random-walk adjacency normalization, multi-head attention GAT, GraphSAGE neighborhood sampling, 1-WL expressive GIN, dynamic EdgeConv, Graph Transformers, Jumping Knowledge aggregation, global pooling readouts, synthetic datasets (Erdős–Rényi planted community, Karate club), and graph explainability.

## Architecture

| Module | Description |
|---|---|
| `graph` | CSR / Edge list graph data structure, degrees, subgraph extraction, uniform neighbor sampling |
| `layers` | GCN, GAT (multi-head attention), GraphSAGE, GIN (learnable $\epsilon$), GatedConv, EdgeConv, Transformer |
| `readout` | `global_add_pool`, `global_mean_pool`, `global_max_pool`, `JumpingKnowledge` (Concat, Max, Last) |
| `models` | High-level `GcnModel`, `GatModel`, `SageModel`, `GinModel`, `EdgeClassifier`, `EdgeRegressor` |
| `datasets` | Random community graphs, cycle graphs, Zachary's Karate Club, mini-batch `GraphLoader` |
| `explain` | Gradient & norm-based node and edge saliency score generation (`ExplanationReport`) |

## Quality & Verification

- **Tests**: 7,179 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-gnn -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
