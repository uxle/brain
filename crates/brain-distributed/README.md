# `brain-distributed`

Pure-Rust distributed training primitives: collectives, communication, parallelism strategies, gradient compression, and fault tolerance.

## Overview

`brain-distributed` implements distributed deep learning infrastructure over `brain-core` tensors with zero external dependencies (no NCCL/MPI). It provides collective communication primitives (ring/tree all-reduce, broadcast), a pluggable `CommBackend` transport abstraction, data/model/tensor/pipeline parallelism, gradient bucketing and Top-K compression, barriers, fault policies, and a simulated multi-rank execution harness for testing.

## Features

- **Collectives**: `CollectiveOp` trait (`allreduce`, `broadcast`), `execute_allreduce` with `AllReduceConfig`/`AllReduceAlgorithm`, `RingTopology`/`TreeTopology`.
- **Communication**: `CommBackend` trait, `MessageHeader` framing, `serialize_tensor`/`deserialize_tensor`.
- **Parallelism**: `DataParallel` gradient sync, `ModelParallelStage`, `TensorParallelLinear` (row/column sharding), `PipelineStage` with 1F1B micro-batching.
- **Gradient optimization**: `GradBucket` bucketing, `topk_compress` sparsification, `split_tensor_for_allreduce`/`concat_chunks`.
- **Coordination**: `DistributedContext` (rank/world size, `is_master`), `ProcessGroup`, `Barrier`, `AsyncCollective` handles.
- **Resilience & ops**: `FaultPolicy` (Retry/FailFast/ExcludeRank), `ClusterNode`/`ClusterTopology`, `CommBench` benchmarks, `run_simulated_rank` harness, `crc32_checksum`.

## Modules

| Module | Contents |
|---|---|
| `core`/`config`/`builder` | `DistributedContext`, `Rank`, `WorldSize`, `DistributedConfig`, `BackendKind`, `DistributedBuilder` |
| `collective` | `CollectiveOp`, all-reduce, ring/tree topologies |
| `comm` | `CommBackend`, `MessageHeader`, tensor serialization |
| `data_parallel`/`model_parallel`/`pipeline`/`tensor_parallel` | parallelism strategies and schedules |
| `grad_allreduce`/`grad_compression` | gradient bucketing, Top-K compression |
| `group`/`sync`/`async_exec` | process groups, barriers, async collectives |
| `fault`/`cluster`/`topology` | failure policies, node management, topology mapping |
| `bench`/`process`/`ops`/`transform`/`utils` | benchmarks, rank simulation, tensor sharding, checksums |

## Quick Start

```rust
use brain_distributed::prelude::*;

let ctx = DistributedContext::new(0, 4);
assert!(ctx.is_master());
```

## Testing

```bash
cargo test -p brain-distributed -j 2
```

## Workspace Role

Depends solely on `brain-core`; `brain-distributed` extends single-device training to multi-rank, multi-node setups with deterministic, in-process simulation of distributed semantics.