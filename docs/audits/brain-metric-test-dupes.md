# Test Duplication & Inflation Audit Report

- **Target Directory:** `crates/brain-metric/src`
- **Total Test Functions Scanned:** 8662
- **Duplicate / Template Groups:** 25
- **Total Padded / Duplicate Test Functions:** 8662 (100.0% of total tests)
- **Redundant Functions Removable:** 8637

## Summary by File

| File | Total Tests | Duplicated Tests | Redundancy Ratio |
|---|---|---|---|
| `crates/brain-metric/src/aggregate.rs` | 366 | 366 | 100.0% |
| `crates/brain-metric/src/classification/auc.rs` | 296 | 296 | 100.0% |
| `crates/brain-metric/src/classification/calibration.rs` | 409 | 409 | 100.0% |
| `crates/brain-metric/src/classification/mod.rs` | 296 | 296 | 100.0% |
| `crates/brain-metric/src/cluster/mod.rs` | 472 | 472 | 100.0% |
| `crates/brain-metric/src/compare.rs` | 474 | 474 | 100.0% |
| `crates/brain-metric/src/config.rs` | 298 | 298 | 100.0% |
| `crates/brain-metric/src/core.rs` | 465 | 465 | 100.0% |
| `crates/brain-metric/src/detection/map.rs` | 327 | 327 | 100.0% |
| `crates/brain-metric/src/detection/mod.rs` | 330 | 330 | 100.0% |
| `crates/brain-metric/src/imbalance.rs` | 369 | 369 | 100.0% |
| `crates/brain-metric/src/impl_.rs` | 328 | 328 | 100.0% |
| `crates/brain-metric/src/multilabel.rs` | 413 | 413 | 100.0% |
| `crates/brain-metric/src/nlp/mod.rs` | 409 | 409 | 100.0% |
| `crates/brain-metric/src/nlp/other.rs` | 253 | 253 | 100.0% |
| `crates/brain-metric/src/ops.rs` | 191 | 191 | 100.0% |
| `crates/brain-metric/src/ranking/mod.rs` | 328 | 328 | 100.0% |
| `crates/brain-metric/src/regression/mod.rs` | 296 | 296 | 100.0% |
| `crates/brain-metric/src/regression/robust.rs` | 299 | 299 | 100.0% |
| `crates/brain-metric/src/report.rs` | 330 | 330 | 100.0% |
| `crates/brain-metric/src/segmentation/mod.rs` | 365 | 365 | 100.0% |
| `crates/brain-metric/src/stats/mod.rs` | 413 | 413 | 100.0% |
| `crates/brain-metric/src/time_series.rs` | 412 | 412 | 100.0% |
| `crates/brain-metric/src/track.rs` | 329 | 329 | 100.0% |
| `crates/brain-metric/src/utils.rs` | 194 | 194 | 100.0% |

## Top Duplicate Groups

### Group 1: 474 identical functions (e.g. `test_compare_stress_001` in `crates/brain-metric/src/compare.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/compare.rs:32`):
```rust
fn test_compare_stress_001() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }
```

### Group 2: 472 identical functions (e.g. `test_cluster_stress_001` in `crates/brain-metric/src/cluster/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/cluster/mod.rs:42`):
```rust
fn test_cluster_stress_001() {
        let clusters = vec![0, 0, 1, 1];
        let labels = vec![10, 10, 20, 20];
        assert_eq!(cluster_purity(&clusters, &labels), 1.0);
    }
```

### Group 3: 465 identical functions (e.g. `test_core_stress_001` in `crates/brain-metric/src/core.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/core.rs:93`):
```rust
fn test_core_stress_001() {
        let v = MetricValue::Scalar(1 as f64 * 0.1);
        assert_eq!(v.as_scalar(), Some(1 as f64 * 0.1));
        assert_eq!(MetricKind::default(), MetricKind::Accuracy);
    }
```

### Group 4: 413 identical functions (e.g. `test_multilabel_stress_001` in `crates/brain-metric/src/multilabel.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/multilabel.rs:46`):
```rust
fn test_multilabel_stress_001() {
        let p = vec![vec![true, false], vec![false, true]];
        let t = vec![vec![true, false], vec![true, true]];
        assert_eq!(exact_match_ratio(&p, &t), 0.5);
        assert_eq!(hamming_loss(&p, &t), 0.25);
    }
```

### Group 5: 413 identical functions (e.g. `test_stats_stress_001` in `crates/brain-metric/src/stats/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/stats/mod.rs:44`):
```rust
fn test_stats_stress_001() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }
```

### Group 6: 412 identical functions (e.g. `test_ts_stress_001` in `crates/brain-metric/src/time_series.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/time_series.rs:52`):
```rust
fn test_ts_stress_001() {
        let f = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(mase_score(&f, &y, 1), 0.0);
        assert_eq!(forecast_bias(&f, &y), 0.0);
    }
```

### Group 7: 409 identical functions (e.g. `test_calib_stress_001` in `crates/brain-metric/src/classification/calibration.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/classification/calibration.rs:74`):
```rust
fn test_calib_stress_001() {
        let confs = vec![0.95, 0.9, 0.8, 0.2];
        let correct = vec![true, true, true, false];
        let report = compute_calibration(&confs, &correct, 10);
        assert!(report.ece >= 0.0 && report.ece <= 1.0);
    }
```

### Group 8: 409 identical functions (e.g. `test_nlp_mod_stress_001` in `crates/brain-metric/src/nlp/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/nlp/mod.rs:73`):
```rust
fn test_nlp_mod_stress_001() {
        let hyp = ["the", "cat", "sat", "on", "the", "mat"];
        let ref_ = ["the", "cat", "sat", "on", "the", "mat"];
        let score = sentence_bleu(&hyp, &ref_, 4);
        assert!(score > 0.9);
    }
```

### Group 9: 369 identical functions (e.g. `test_imbalance_stress_001` in `crates/brain-metric/src/imbalance.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/imbalance.rs:29`):
```rust
fn test_imbalance_stress_001() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }
```

### Group 10: 366 identical functions (e.g. `test_aggregate_stress_001` in `crates/brain-metric/src/aggregate.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/aggregate.rs:51`):
```rust
fn test_aggregate_stress_001() {
        let runs = vec![0.85, 0.86, 0.84, 0.85];
        let agg = aggregate_metric_runs(&runs);
        assert!((agg.mean - 0.85).abs() < 1e-9);
        assert!(agg.ci_95_lower <= agg.mean);
        assert!(agg.ci_95_upper >= agg.mean);
    }
```

### Group 11: 365 identical functions (e.g. `test_seg_mod_stress_001` in `crates/brain-metric/src/segmentation/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/segmentation/mod.rs:62`):
```rust
fn test_seg_mod_stress_001() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 1, 0];
        let (miou, acc) = miou_and_pixel_accuracy(&p, &t, 2);
        assert_eq!(miou, 1.0);
        assert_eq!(acc, 1.0);
    }
```

### Group 12: 330 identical functions (e.g. `test_report_stress_001` in `crates/brain-metric/src/report.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/report.rs:49`):
```rust
fn test_report_stress_001() {
        let mut m = BTreeMap::new();
        m.insert("accuracy".into(), 0.95);
        let md = format_markdown_report(&m);
        assert!(md.contains("| `accuracy` | 0.9500 |"));
        let csv = format_csv_report(&m);
        assert!(csv.contains("accuracy,0.95"));
    }
```

### Group 13: 330 identical functions (e.g. `test_det_mod_stress_001` in `crates/brain-metric/src/detection/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/detection/mod.rs:42`):
```rust
fn test_det_mod_stress_001() {
        let b1 = [0.0, 0.0, 10.0, 10.0];
        let b2 = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(bbox_iou(&b1, &b2), 1.0);

        let b3 = [20.0, 20.0, 30.0, 30.0];
        assert_eq!(bbox_iou(&b1, &b3), 0.0);
    }
```

### Group 14: 329 identical functions (e.g. `test_track_stress_001` in `crates/brain-metric/src/track.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/track.rs:56`):
```rust
fn test_track_stress_001() {
        let mut tracker = MetricTracker::new();
        tracker.update("loss", 0.5, 2);
        tracker.update("loss", 0.3, 2);
        assert!((tracker.mean("loss").unwrap() - 0.4).abs() < 1e-9);
        tracker.reset();
        assert!(tracker.mean("loss").is_none());
    }
```

### Group 15: 328 identical functions (e.g. `test_impl_stress_001` in `crates/brain-metric/src/impl_.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/impl_.rs:61`):
```rust
fn test_impl_stress_001() {
        let p = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let t = Tensor::from_vec(vec![1.0, 0.0], vec![2]);
        let cfg = default_config(MetricKind::Accuracy);
        let res = compute_metric(MetricKind::Accuracy, &p, &t, &cfg).unwrap();
        assert_eq!(res.as_scalar(), Some(1.0));
        assert!(metric_names().contains(&"Accuracy"));
    }
```

### Group 16: 328 identical functions (e.g. `test_ranking_stress_001` in `crates/brain-metric/src/ranking/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/ranking/mod.rs:65`):
```rust
fn test_ranking_stress_001() {
        let hits = vec![vec![false, true, false]]; // First relevant item is at rank 2
        let mrr = mean_reciprocal_rank(&hits);
        assert_eq!(mrr, 0.5);

        let ndcg = ndcg_at_k(&[3.0, 2.0, 3.0, 0.0], 3);
        assert!(ndcg > 0.8 && ndcg <= 1.0);
    }
```

### Group 17: 327 identical functions (e.g. `test_map_stress_001` in `crates/brain-metric/src/detection/map.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/detection/map.rs:79`):
```rust
fn test_map_stress_001() {
        let p = vec![[0.0, 0.0, 10.0, 10.0]];
        let s = vec![0.9];
        let g = vec![[0.0, 0.0, 10.0, 10.0]];
        let cfg = MapConfig::default();
        let map = mean_average_precision(&p, &s, &g, &cfg);
        assert_eq!(map, 1.0);
    }
```

### Group 18: 299 identical functions (e.g. `test_robust_stress_001` in `crates/brain-metric/src/regression/robust.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/regression/robust.rs:57`):
```rust
fn test_robust_stress_001() {
        let p = Tensor::from_vec(vec![1.0, 2.0, 10.0], vec![3]);
        let t = Tensor::from_vec(vec![1.0, 2.0, 2.0], vec![3]);
        let med_ae = median_absolute_error(&p, &t);
        assert_eq!(med_ae, 0.0);

        let hm = huber_metric(&p, &t, 1.0);
        assert!(hm > 0.0);
    }
```

### Group 19: 298 identical functions (e.g. `test_config_stress_001` in `crates/brain-metric/src/config.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/config.rs:68`):
```rust
fn test_config_stress_001() {
        let mut cfg = MetricConfig::default();
        cfg.top_k = 2;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.top_k = 0;
        assert!(cfg.validate().is_err());
    }
```

### Group 20: 296 identical functions (e.g. `test_class_mod_stress_001` in `crates/brain-metric/src/classification/mod.rs`)
- Files involved: 1
- Sample definition (`crates/brain-metric/src/classification/mod.rs:93`):
```rust
fn test_class_mod_stress_001() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }
```
