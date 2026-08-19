# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-dataset/src`
- **Total Test Functions Scanned:** 14675
- **Duplicate / Template Groups:** 34
- **Total Padded / Duplicate Test Functions:** 14675 (100.0% of total tests)
- **Redundant Functions Removable:** 14641

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-dataset/src/analyze.rs` | 553 | 553 | 100.0% |
| `crates/brain-dataset/src/balance.rs` | 553 | 553 | 100.0% |
| `crates/brain-dataset/src/builder.rs` | 471 | 471 | 100.0% |
| `crates/brain-dataset/src/cache.rs` | 413 | 413 | 100.0% |
| `crates/brain-dataset/src/compute.rs` | 474 | 474 | 100.0% |
| `crates/brain-dataset/src/config.rs` | 473 | 473 | 100.0% |
| `crates/brain-dataset/src/core.rs` | 365 | 365 | 100.0% |
| `crates/brain-dataset/src/dataset/audio.rs` | 412 | 412 | 100.0% |
| `crates/brain-dataset/src/dataset/mod.rs` | 206 | 206 | 100.0% |
| `crates/brain-dataset/src/dataset/tabular.rs` | 366 | 366 | 100.0% |
| `crates/brain-dataset/src/dataset/text.rs` | 413 | 413 | 100.0% |
| `crates/brain-dataset/src/dataset/vision.rs` | 412 | 412 | 100.0% |
| `crates/brain-dataset/src/dataset/vision_v2.rs` | 411 | 411 | 100.0% |
| `crates/brain-dataset/src/helper.rs` | 475 | 475 | 100.0% |
| `crates/brain-dataset/src/impl.rs` | 219 | 219 | 100.0% |
| `crates/brain-dataset/src/inspect.rs` | 220 | 220 | 100.0% |
| `crates/brain-dataset/src/lib.rs` | 404 | 404 | 100.0% |
| `crates/brain-dataset/src/loaders/mod.rs` | 206 | 206 | 100.0% |
| `crates/brain-dataset/src/loaders/worker.rs` | 554 | 554 | 100.0% |
| `crates/brain-dataset/src/manage.rs` | 473 | 473 | 100.0% |
| `crates/brain-dataset/src/ops.rs` | 475 | 475 | 100.0% |
| `crates/brain-dataset/src/optimize.rs` | 553 | 553 | 100.0% |
| `crates/brain-dataset/src/process.rs` | 554 | 554 | 100.0% |
| `crates/brain-dataset/src/samplers/mod.rs` | 550 | 550 | 100.0% |
| `crates/brain-dataset/src/splits.rs` | 414 | 414 | 100.0% |
| `crates/brain-dataset/src/statistics.rs` | 553 | 553 | 100.0% |
| `crates/brain-dataset/src/stream.rs` | 472 | 472 | 100.0% |
| `crates/brain-dataset/src/transform.rs` | 413 | 413 | 100.0% |
| `crates/brain-dataset/src/transforms/audio_t.rs` | 414 | 414 | 100.0% |
| `crates/brain-dataset/src/transforms/mod.rs` | 411 | 411 | 100.0% |
| `crates/brain-dataset/src/transforms/numeric_t.rs` | 414 | 414 | 100.0% |
| `crates/brain-dataset/src/transforms/text_t.rs` | 413 | 413 | 100.0% |
| `crates/brain-dataset/src/transforms/vision_t.rs` | 414 | 414 | 100.0% |
| `crates/brain-dataset/src/utils.rs` | 552 | 552 | 100.0% |

## Top Duplicate Groups

### Group 1: 554 identical functions (e.g. `test_process_stress_001` in `crates/brain-dataset/src/process.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/process.rs:26`):
```rust
fn test_process_stress_001() {
        let p = ProcessConfig::default();
        assert_eq!(p.thread_count, 4);
    }
```

### Group 2: 554 identical functions (e.g. `test_worker_stress_001` in `crates/brain-dataset/src/loaders/worker.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/loaders/worker.rs:26`):
```rust
fn test_worker_stress_001() {
        let wp = WorkerPool::new(4);
        assert_eq!(wp.num_workers, 4);
    }
```

### Group 3: 553 identical functions (e.g. `test_analyze_stress_001` in `crates/brain-dataset/src/analyze.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/analyze.rs:27`):
```rust
fn test_analyze_stress_001() {
        let a = AnalysisReport::new(100);
        assert_eq!(a.total_samples, 100);
    }
```

### Group 4: 553 identical functions (e.g. `test_optimize_stress_001` in `crates/brain-dataset/src/optimize.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/optimize.rs:27`):
```rust
fn test_optimize_stress_001() {
        let r = OptimizeReport::new(4);
        assert_eq!(r.suggested_num_workers, 4);
    }
```

### Group 5: 553 identical functions (e.g. `test_statistics_stress_001` in `crates/brain-dataset/src/statistics.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/statistics.rs:31`):
```rust
fn test_statistics_stress_001() {
        let s = DatasetStats::new(1000, 10);
        assert_eq!(s.total_samples, 1000);
    }
```

### Group 6: 553 identical functions (e.g. `test_balance_stress_001` in `crates/brain-dataset/src/balance.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/balance.rs:27`):
```rust
fn test_balance_stress_001() {
        let b = BalanceConfig::new(100);
        assert_eq!(b.target_samples_per_class, 100);
    }
```

### Group 7: 552 identical functions (e.g. `test_utils_stress_001` in `crates/brain-dataset/src/utils.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/utils.rs:34`):
```rust
fn test_utils_stress_001() {
        let mut rng = DatasetRng::new(1);
        assert_ne!(rng.next_u64(), 0);
    }
```

### Group 8: 550 identical functions (e.g. `test_samplers_mod_stress_001` in `crates/brain-dataset/src/samplers/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/samplers/mod.rs:45`):
```rust
fn test_samplers_mod_stress_001() {
        let s = SequentialSampler::new(1);
        assert_eq!(s.len(), 1);
    }
```

### Group 9: 475 identical functions (e.g. `test_helper_stress_001` in `crates/brain-dataset/src/helper.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/helper.rs:21`):
```rust
fn test_helper_stress_001() {
        let t = Tensor::zeros(vec![3, 32, 32]);
        let s = format_tensor_shape(&t);
        assert!(s.contains("3"));
    }
```

### Group 10: 475 identical functions (e.g. `test_ops_stress_001` in `crates/brain-dataset/src/ops.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/ops.rs:24`):
```rust
fn test_ops_stress_001() {
        let b = Batch::new(vec![Item::new(1, Tensor::zeros(vec![1]))]);
        let b2 = map_batch(b, |it| Item::new(it.id + 1, it.data));
        assert_eq!(b2.items[0].id, 1 + 1);
    }
```

### Group 11: 474 identical functions (e.g. `test_compute_stress_001` in `crates/brain-dataset/src/compute.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/compute.rs:26`):
```rust
fn test_compute_stress_001() {
        let b = Batch::new(vec![Item::new(1, Tensor::zeros(vec![2, 2]))]);
        let mean = compute_batch_mean(&b);
        assert_eq!(mean.shape(), &[2, 2]);
    }
```

### Group 12: 473 identical functions (e.g. `test_config_stress_001` in `crates/brain-dataset/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/config.rs:34`):
```rust
fn test_config_stress_001() {
        let cfg = DatasetConfig::default();
        assert_eq!(cfg.batch_size, 32);
        assert!(!cfg.shuffle);
    }
```

### Group 13: 473 identical functions (e.g. `test_manage_stress_001` in `crates/brain-dataset/src/manage.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/manage.rs:39`):
```rust
fn test_manage_stress_001() {
        let mut reg = DatasetRegistry::new();
        reg.register("mnist", 60000);
        assert_eq!(reg.lookup("mnist"), Some(60000));
    }
```

### Group 14: 472 identical functions (e.g. `test_stream_stress_001` in `crates/brain-dataset/src/stream.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/stream.rs:42`):
```rust
fn test_stream_stress_001() {
        let mut r = StreamingReader::new();
        let it = r.next_item().unwrap();
        assert_eq!(it.id, 0);
    }
```

### Group 15: 471 identical functions (e.g. `test_builder_stress_001` in `crates/brain-dataset/src/builder.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/builder.rs:53`):
```rust
fn test_builder_stress_001() {
        let cfg = DatasetBuilder::new().batch_size(64).shuffle(true).build();
        assert_eq!(cfg.batch_size, 64);
        assert!(cfg.shuffle);
    }
```

### Group 16: 414 identical functions (e.g. `test_splits_stress_001` in `crates/brain-dataset/src/splits.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/splits.rs:37`):
```rust
fn test_splits_stress_001() {
        let res = random_split_indices(100, 0.7, 0.15);
        assert_eq!(res.train_indices.len(), 70);
        assert_eq!(res.val_indices.len(), 15);
        assert_eq!(res.test_indices.len(), 15);
    }
```

### Group 17: 414 identical functions (e.g. `test_vision_t_stress_001` in `crates/brain-dataset/src/transforms/vision_t.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/transforms/vision_t.rs:36`):
```rust
fn test_vision_t_stress_001() {
        let norm = Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]);
        let item = Item::new(1, Tensor::zeros(vec![3, 32, 32]));
        let out = norm.apply(item);
        assert_eq!(out.id, 1);
    }
```

### Group 18: 414 identical functions (e.g. `test_audio_t_stress_001` in `crates/brain-dataset/src/transforms/audio_t.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/transforms/audio_t.rs:36`):
```rust
fn test_audio_t_stress_001() {
        let r = Resample::new(44100, 16000);
        let item = Item::new(1, Tensor::zeros(vec![1, 44100]));
        let out = r.apply(item);
        assert_eq!(out.id, 1);
    }
```

### Group 19: 414 identical functions (e.g. `test_numeric_t_stress_001` in `crates/brain-dataset/src/transforms/numeric_t.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/transforms/numeric_t.rs:36`):
```rust
fn test_numeric_t_stress_001() {
        let s = MinMaxScale::new(0.0, 1.0);
        let item = Item::new(1, Tensor::zeros(vec![10]));
        let out = s.apply(item);
        assert_eq!(out.id, 1);
    }
```

### Group 20: 413 identical functions (e.g. `test_transform_stress_001` in `crates/brain-dataset/src/transform.rs`)
- Files involved: 1
- Sample definition (`crates/brain-dataset/src/transform.rs:40`):
```rust
fn test_transform_stress_001() {
        let g = TransformGraph::new().add_stage();
        let it = Item::new(1, Tensor::zeros(vec![1]));
        let out = g.execute(it);
        assert_eq!(out.id, 1);
    }
```
