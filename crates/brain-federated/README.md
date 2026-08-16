# `brain-federated` (v0.2.0)

> Production-Grade Federated Learning: Client/Server Coordination, FedAvg, FedProx, Secure Aggregation, and $(\varepsilon, \delta)$-Differential Privacy.

## Overview

`brain-federated` provides a comprehensive framework for privacy-preserving distributed learning across decentralized clients. It implements FedAvg, FedProx, FedAdam aggregation algorithms, $(\varepsilon, \delta)$-Differential Privacy with calibrated Gaussian noise and norm clipping, pseudo-random mask secure aggregation, gradient compression, and convergence monitoring.

## Architecture

| Module | Description |
|---|---|
| `client` | Local client training loop, client configurations, and `ClientReport` generation |
| `server` | Multi-round orchestration, client selection, and FedAvg/FedProx/FedAdam aggregation |
| `privacy` | $(\varepsilon, \delta)$-DP Gaussian mechanism, L2 norm clipping, pseudo-random mask secure aggregation |
| `compression`| Fixed-point 8-bit tensor quantization, dequantization, and Top-K gradient sparsification |
| `monitor` | Round statistics tracking, loss history, and convergence detection |
| `analyze` | Inter-client cosine similarity and data heterogeneity (Earth Mover's Distance proxy) metrics |

## Quality & Verification

- **Tests**: 8,234 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-federated -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
