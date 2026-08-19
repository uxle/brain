# `brain-dataset`

Dataset ecosystem for vision, text, audio, and tabular modalities: generators, transforms, samplers, loaders, splits, caching, and statistics.

## Overview

`brain-dataset` provides the high-level dataset layer of the Brain framework over `brain-core` tensors with zero external dependencies. It defines the `Dataset` trait with `Item`/`Batch`, ships ready-made datasets (tabular, text lines, random images, segmentation, synthetic audio), a multi-worker `DataLoader` and `WorkerPool`, transform pipelines, samplers, train/val/test splitting, caching, and a fluent `DatasetBuilder`.

## Features

- **Datasets**: `TabularDataset`, `TextLinesDataset`, `RandomImageDataset`, `RandomSegDataset`, `SyntheticAudioDataset`, plus `Subset` views over any `Dataset`.
- **Loading**: `DataLoader` with batching (`fetch_batch`), `WorkerPool` for multi-worker iteration.
- **Transforms**: `Transform` trait with `Compose`, `Normalize`, `PadOrTruncate`, `MinMaxScale`, `Resample`, and `TransformGraph` pipelines.
- **Samplers & splits**: `Sampler` trait + `SequentialSampler`, `random_split_indices`/`SplitResult`.
- **Management**: `DatasetRegistry` (named registrations), `DatasetCache`, `DatasetStats`, `InspectionReport`, `BalanceConfig` rebalancing, `OptimizeReport`.
- **Streaming & utils**: `StreamingReader` (item-at-a-time), `compute_batch_mean`, `map_batch`, `DatasetRng`, `format_tensor_shape`.
- **Builder**: `DatasetBuilder` (batch size, shuffle) producing a `DatasetConfig`.

## Modules

| Module | Contents |
|---|---|
| `core`/`dataset` | `Item`, `Batch`, `Dataset` trait, concrete dataset generators, `Subset` |
| `loaders`/`samplers` | `DataLoader`, `WorkerPool`, `SequentialSampler` |
| `transforms`/`transform` | `Transform` trait, `Compose`, domain transforms, `TransformGraph` |
| `splits`/`stream` | train/val/test splitting, `StreamingReader` |
| `cache`/`manage`/`balance`/`statistics`/`inspect`/`optimize` | caching, registry, rebalancing, stats, inspection, worker tuning |
| `builder`/`config`/`compute`/`ops`/`helper`/`utils` | builder, config, batch math, mapping, formatting, RNG |

## Quick Start

```rust
use brain_core::Tensor;
use brain_dataset::dataset::tabular::TabularDataset;
use brain_dataset::loaders::DataLoader;

let features = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let labels = Tensor::from_slice(&[0.0, 1.0], vec![2]);
let ds = TabularDataset::new(features, Some(labels));

let loader = DataLoader::new(&ds, 2);
let batch = loader.fetch_batch().expect("batch");
assert_eq!(batch.len(), 2);
```

## Testing

```bash
cargo test -p brain-dataset -j 2
```

## Workspace Role

Depends solely on `brain-core`; `brain-dataset` sits on top of the raw pipeline layer (`brain-data`) to offer ready-made dataset generators, transforms, and loaders for training loops across the Brain framework.