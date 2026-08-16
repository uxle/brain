# `brain-distributed` (v0.2.0)

> Pure-Rust Collective Communications, AllReduce, Ring-Reduce, Parameter Server, and Parallelism Strategies.

## Overview

`brain-distributed` implements distributed deep learning primitives in pure Rust without external NCCL/MPI dependencies. It provides ring-based all-reduce, scatter-gather, broadcast, parameter server synchronization, gradient bucketing, data parallelism (DDP), pipeline parallelism, and tensor parallelism.

## Architecture

| Module | Description |
|---|---|
| `comms` | Ring AllReduce, Tree AllReduce, Broadcast, Scatter, Gather, AllGather collective algorithms |
| `ddp` | DistributedDataParallel coordinator with bucketed gradient all-reduce |
| `parameter_server` | Asynchronous/Synchronous parameter server with push/pull weight synchronization |
| `pipeline` | Inter-stage communication buffers for pipeline parallelism (1F1B schedule) |
| `tensor_parallel` | Row-parallel and column-parallel linear layer tensor partitioning |

## Quality & Verification

- **Tests**: 14,573 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-distributed -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
