# `brain-federated`

Pure-Rust federated learning: server/client orchestration, FedAvg aggregation, differential privacy, secure aggregation, and gradient compression.

## Overview

`brain-federated` provides privacy-preserving distributed learning over `brain-core` tensors with zero external dependencies. It implements a `FederatedServer`/client architecture with FedAvg-style weighted aggregation, Gaussian differential-privacy noise, mask-based secure aggregation, quantization and Top-K sparsification of client updates, client sampling, and convergence monitoring.

## Features

- **Server side**: `FederatedServer` (round orchestration, `advance_round`), `fed_avg_aggregate`, `AggregationAlgorithm`, `RoundStats`, `ServerConfig`.
- **Client side**: `LocalTrainer` trait with `SgdTrainer`, `ClientConfig`/`ClientReport`, local training loop.
- **Privacy**: `GaussianNoise`/`DpConfig`, `add_dp_noise`, `SecureAggregator` with `mask_tensor` masks.
- **Compression**: `QuantConfig`/`SparseConfig`, `quantize_tensor`/`dequantize_tensor`, `top_k_sparsify`.
- **Aggregation math**: `ModelDelta` (client id, weights, sample counts), `scale_delta`, `l2_norm_delta`, `multiply_accumulate`.
- **Optimization helpers**: `clip_grad_norm`/`global_grad_norm`, `cosine_lr`, `apply_weight_decay`, `polyak_average`, `normalize_weights`, `mse_eval`.
- **Monitoring & analysis**: `FedMonitor`, `FedSystemBuilder` (num_clients, rounds, fraction_fit, local_epochs), `sample_clients`, `estimate_heterogeneity`, `communication_cost_bytes`, `cosine_similarity_deltas`.

## Modules

| Module | Contents |
|---|---|
| `core`/`config` | `ClientId`, `ModelDelta`, `ClientMetrics`, `ServerMetrics`, `RoundId`, `FedConfig` |
| `server`/`client` | round orchestration, aggregation, local training |
| `privacy` | Gaussian DP noise, secure aggregation masks |
| `compression` | quantization, Top-K sparsification |
| `monitor`/`analyze` | convergence tracking, heterogeneity/communication analysis |
| `process`/`transform`/`compute`/`ops` | LR schedules, weight decay, Polyak averaging, clipping, delta ops |
| `builder`/`utils`/`impl_` | `FedSystemBuilder`, `sample_clients`, `stddev`, `run_round` |

## Quick Start

```rust
use brain_core::Tensor;
use brain_federated::{fed_avg_aggregate, ModelDelta};

let deltas = vec![
    ModelDelta::new(0, vec![Tensor::from_slice(&[1.0, 2.0], vec![2])], 4),
    ModelDelta::new(1, vec![Tensor::from_slice(&[3.0, 4.0], vec![2])], 6),
];
let averaged = fed_avg_aggregate(&deltas);
```

## Testing

```bash
cargo test -p brain-federated -j 2
```

## Workspace Role

Depends solely on `brain-core`; `brain-federated` layers privacy-preserving, decentralized training workflows on top of the Brain framework's tensor stack.