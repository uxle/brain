# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-data/src`
- **Total Test Functions Scanned:** 11035
- **Duplicate / Template Groups:** 26
- **Total Padded / Duplicate Test Functions:** 11035 (100.0% of total tests)
- **Redundant Functions Removable:** 11009

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-data/src/backpressure.rs` | 553 | 553 | 100.0% |
| `crates/brain-data/src/batch.rs` | 235 | 235 | 100.0% |
| `crates/brain-data/src/caching.rs` | 413 | 413 | 100.0% |
| `crates/brain-data/src/checkpoint.rs` | 474 | 474 | 100.0% |
| `crates/brain-data/src/collate.rs` | 474 | 474 | 100.0% |
| `crates/brain-data/src/compression.rs` | 367 | 367 | 100.0% |
| `crates/brain-data/src/config.rs` | 474 | 474 | 100.0% |
| `crates/brain-data/src/core.rs` | 326 | 326 | 100.0% |
| `crates/brain-data/src/errors.rs` | 552 | 552 | 100.0% |
| `crates/brain-data/src/impl.rs` | 220 | 220 | 100.0% |
| `crates/brain-data/src/interop.rs` | 475 | 475 | 100.0% |
| `crates/brain-data/src/lazy.rs` | 472 | 472 | 100.0% |
| `crates/brain-data/src/lib.rs` | 402 | 402 | 100.0% |
| `crates/brain-data/src/loading.rs` | 472 | 472 | 100.0% |
| `crates/brain-data/src/metrics.rs` | 368 | 368 | 100.0% |
| `crates/brain-data/src/mmap.rs` | 552 | 552 | 100.0% |
| `crates/brain-data/src/multi.rs` | 220 | 220 | 100.0% |
| `crates/brain-data/src/ops.rs` | 473 | 473 | 100.0% |
| `crates/brain-data/src/pipeline.rs` | 414 | 414 | 100.0% |
| `crates/brain-data/src/prefetch.rs` | 330 | 330 | 100.0% |
| `crates/brain-data/src/profile.rs` | 553 | 553 | 100.0% |
| `crates/brain-data/src/samplers.rs` | 409 | 409 | 100.0% |
| `crates/brain-data/src/shuffle.rs` | 554 | 554 | 100.0% |
| `crates/brain-data/src/stages.rs` | 412 | 412 | 100.0% |
| `crates/brain-data/src/streaming.rs` | 473 | 473 | 100.0% |
| `crates/brain-data/src/utils.rs` | 368 | 368 | 100.0% |

## Top Duplicate Groups

### Group 1: 554 identical functions (e.g. `test_shuffle_stress_001` in `crates/brain-data/src/shuffle.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/shuffle.rs:26`):
```rust
fn test_shuffle_stress_001() {
        let perm = shuffle_indices(10, 1);
        assert_eq!(perm.len(), 10);
    }
```

### Group 2: 553 identical functions (e.g. `test_profile_stress_001` in `crates/brain-data/src/profile.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/profile.rs:31`):
```rust
fn test_profile_stress_001() {
        let p = StageProfile::new("map_stage", Duration::from_millis(5));
        assert_eq!(p.name, "map_stage");
    }
```

### Group 3: 553 identical functions (e.g. `test_backpressure_stress_001` in `crates/brain-data/src/backpressure.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/backpressure.rs:28`):
```rust
fn test_backpressure_stress_001() {
        let cfg = BackpressureConfig::default();
        assert_eq!(cfg.max_buffered_batches, 16);
    }
```

### Group 4: 552 identical functions (e.g. `test_pipeline_errors_stress_001` in `crates/brain-data/src/errors.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/errors.rs:34`):
```rust
fn test_pipeline_errors_stress_001() {
        let err = PipelineError::CorruptSample(format!("sample_1"));
        assert!(format!("{}", err).contains("Corrupt sample:"));
    }
```

### Group 5: 552 identical functions (e.g. `test_mmap_reader_stress_001` in `crates/brain-data/src/mmap.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/mmap.rs:33`):
```rust
fn test_mmap_reader_stress_001() {
        let r = MmapChunkReader::from_bytes(vec![0; 100]);
        assert_eq!(r.read_slice(0, 10).unwrap().len(), 10);
    }
```

### Group 6: 475 identical functions (e.g. `test_interop_stress_001` in `crates/brain-data/src/interop.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/interop.rs:23`):
```rust
fn test_interop_stress_001() {
        let t = Tensor::zeros(vec![4, 2]);
        let samples = tensor_to_samples(&t);
        assert_eq!(samples.len(), 4);
    }
```

### Group 7: 474 identical functions (e.g. `test_collate_stress_001` in `crates/brain-data/src/collate.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/collate.rs:29`):
```rust
fn test_collate_stress_001() {
        let s = Sample::new(1, Tensor::zeros(vec![2]));
        let b = default_collate(&[s]);
        assert_eq!(b.len(), 1);
    }
```

### Group 8: 474 identical functions (e.g. `test_pipeline_checkpoint_stress_001` in `crates/brain-data/src/checkpoint.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/checkpoint.rs:26`):
```rust
fn test_pipeline_checkpoint_stress_001() {
        let cp = PipelineCheckpoint::new(1, 1);
        assert_eq!(cp.epoch, 1);
        assert_eq!(cp.sample_offset, 1);
    }
```

### Group 9: 474 identical functions (e.g. `test_data_config_stress_001` in `crates/brain-data/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/config.rs:32`):
```rust
fn test_data_config_stress_001() {
        let cfg = DataLoaderConfig::default();
        assert_eq!(cfg.batch_size, 32);
        assert_eq!(cfg.num_workers, 4);
    }
```

### Group 10: 473 identical functions (e.g. `test_data_ops_stress_001` in `crates/brain-data/src/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/ops.rs:36`):
```rust
fn test_data_ops_stress_001() {
        let s = Sample::new(1, Tensor::zeros(vec![2, 2]));
        let s2 = transform_sample_tensor(s, |t| &t + &Tensor::scalar(1.0));
        assert_eq!(s2.id, 1);
    }
```

### Group 11: 473 identical functions (e.g. `test_streaming_stress_001` in `crates/brain-data/src/streaming.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/streaming.rs:36`):
```rust
fn test_streaming_stress_001() {
        let mut ds = StreamDataset::new(10);
        let s = ds.next_sample(0).unwrap();
        assert_eq!(s.id, 0);
    }
```

### Group 12: 472 identical functions (e.g. `test_loading_stress_001` in `crates/brain-data/src/loading.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/loading.rs:42`):
```rust
fn test_loading_stress_001() {
        let loader = MemoryLoader::from_tensors(vec![Tensor::zeros(vec![2, 2])]);
        assert_eq!(loader.len(), 1);
        assert!(loader.get(0).is_some());
    }
```

### Group 13: 472 identical functions (e.g. `test_lazy_sample_stress_001` in `crates/brain-data/src/lazy.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/lazy.rs:42`):
```rust
fn test_lazy_sample_stress_001() {
        let lazy = LazySample::new(|| Sample::new(1, Tensor::zeros(vec![1])));
        let s = lazy.evaluate();
        assert_eq!(s.id, 1);
    }
```

### Group 14: 414 identical functions (e.g. `test_pipeline_stress_001` in `crates/brain-data/src/pipeline.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/pipeline.rs:38`):
```rust
fn test_pipeline_stress_001() {
        let p = Pipeline::new().add_stage();
        let b = SampleBatch::new(vec![crate::core::Sample::new(1, Tensor::zeros(vec![1]))]);
        let out = p.process_batch(b);
        assert_eq!(out.len(), 1);
    }
```

### Group 15: 413 identical functions (e.g. `test_caching_stress_001` in `crates/brain-data/src/caching.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/caching.rs:45`):
```rust
fn test_caching_stress_001() {
        let mut c = SampleCache::new(5);
        let s = Sample::new(1, Tensor::zeros(vec![1]));
        c.put(s);
        assert!(c.get(1).is_some());
    }
```

### Group 16: 412 identical functions (e.g. `test_stages_stress_001` in `crates/brain-data/src/stages.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/stages.rs:52`):
```rust
fn test_stages_stress_001() {
        let stage = MapStage::new("identity", |s| s);
        let s = Sample::new(1, Tensor::zeros(vec![1]));
        let out = stage.process(s).unwrap();
        assert_eq!(out.id, 1);
    }
```

### Group 17: 409 identical functions (e.g. `test_samplers_stress_001` in `crates/brain-data/src/samplers.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/samplers.rs:71`):
```rust
fn test_samplers_stress_001() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }
```

### Group 18: 402 identical functions (e.g. `test_data_lib_stress_001` in `crates/brain-data/src/lib.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/lib.rs:128`):
```rust
fn test_data_lib_stress_001() {
        let v = version_tuple();
        assert_eq!(v, (0, 2, 0));
        let s = version_string();
        assert!(s.contains("0.2.0"));
    }
```

### Group 19: 368 identical functions (e.g. `test_data_utils_stress_001` in `crates/brain-data/src/utils.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/utils.rs:33`):
```rust
fn test_data_utils_stress_001() {
        let h = fnv_hash_bytes(b"sample_key");
        assert_ne!(h, 0);
        let items = vec![1, 2, 2, 3, 1];
        let d = dedup_items(&items);
        assert_eq!(d, vec![1, 2, 3]);
    }
```

### Group 20: 368 identical functions (e.g. `test_pipeline_metrics_stress_001` in `crates/brain-data/src/metrics.rs`)
- Files involved: 1
- Sample definition (`crates/brain-data/src/metrics.rs:33`):
```rust
fn test_pipeline_metrics_stress_001() {
        let m = PipelineMetrics {
            items_processed: 1000,
            elapsed: Duration::from_secs(2),
        };
        assert_eq!(m.throughput(), 500.0);
    }
```
