# `brain-data` (v0.2.0)

> High-Throughput Data Pipelines, Streaming Iterators, Batching, Collation, and Prefetching.

## Overview

`brain-data` manages dataset streaming, prefetching, multi-threaded batch collation, chunked file readers, and tensor packing. It enables seamless feeding of arbitrary data types into training and inference pipelines.

## Architecture

| Module | Description |
|---|---|
| `pipeline` | Composable stream pipelines: map, filter, batch, window, zip, take, skip |
| `collate` | Automatic tensor collation, padding strategies, variable-length sequence collation |
| `prefetch` | Multi-buffered asynchronous prefetcher decoupling I/O from computation |
| `sampler` | Sequential, random, weighted, and distributed shard samplers |

## Quality & Verification

- **Tests**: 11,039 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-data -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
