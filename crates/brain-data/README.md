# `brain-data`

High-throughput data pipeline primitives: sources, samplers, batching, collation, prefetching, caching, streaming, and compression.

## Overview

`brain-data` provides the ingestion layer of the Brain framework over `brain-core` tensors with zero external dependencies. It defines `Sample`/`SampleBatch` and the `DataSource`/`DataReader` traits, offers memory and stream-based loaders, samplers (sequential and distributed), batch iteration with collation, pipeline stages with backpressure and prefetching, capacity-bounded caching, and pure-Rust compression codecs.

## Features

- **Core types**: `Sample`, `SampleBatch`, `DataSource`/`DataReader` traits, `DataLoaderConfig` (batch size, workers, shuffle, drop-last).
- **Loading & streaming**: `MemoryLoader::from_tensors`, `StreamDataset` for chunked corpora, `MmapChunkReader` for byte-buffer chunk reading.
- **Batching & collation**: `BatchIter` with drop-last, `default_collate`/`pad_collate`, `CollateFn` trait.
- **Samplers**: `SequentialSampler` and `DistributedSampler` (disjoint per-replica index splits).
- **Pipelines**: `Pipeline` (add stages, `process_batch`), `PipelineRunner::run_epoch`, `MapStage` transforms, `BackpressureConfig` watermarks, `PrefetchIter`.
- **Caching & shuffle**: `SampleCache` (capacity-bounded, id-keyed), `shuffle_indices` seeded permutation.
- **Compression & utilities**: `rle_encode`/`delta_encode`, `fnv_hash_bytes`, `dedup_items`, `tensor_to_samples` interop.
- **Observability**: `PipelineMetrics`, `StageProfile`, `PipelineCheckpoint`, typed `PipelineError` handling.

## Modules

| Module | Contents |
|---|---|
| `core`/`config` | `Sample`, `SampleBatch`, `DataSource`, `DataReader`, `DataLoaderConfig` |
| `loading`/`streaming`/`mmap` | memory loaders, chunked streaming, mmap readers |
| `samplers`/`batch`/`collate` | sequential/distributed samplers, `BatchIter`, collation functions |
| `pipeline`/`stages`/`prefetch`/`backpressure` | stage pipelines, `MapStage`, prefetch buffers, flow control |
| `caching`/`shuffle`/`checkpoint` | bounded sample cache, permutation shuffle, epoch state recovery |
| `compression`/`multi`/`lazy`/`interop` | RLE/Delta codecs, source concatenation, lazy samples, tensor interop |
| `metrics`/`profile`/`errors`/`ops`/`utils` | throughput tracking, stage profiling, errors, sample transforms |

## Quick Start

```rust
use brain_core::Tensor;
use brain_data::collate::default_collate;
use brain_data::loading::MemoryLoader;
use brain_data::samplers::SequentialSampler;

let loader = MemoryLoader::from_tensors(vec![
    Tensor::from_slice(&[1.0, 2.0], vec![2]),
    Tensor::from_slice(&[3.0, 4.0], vec![2]),
]);

let idx = SequentialSampler::new(loader.len()).sample_indices();
let samples: Vec<_> = idx.iter().filter_map(|&i| loader.get(i)).collect();
let batch = default_collate(&samples);
assert_eq!(batch.len(), 2);
```

## Testing

```bash
cargo test -p brain-data -j 2
```

## Workspace Role

Depends solely on `brain-core`; `brain-data` is the raw-data ingestion and pipeline layer, while higher-level dataset abstractions (generators, transforms, `DataLoader`) live in `brain-dataset`.