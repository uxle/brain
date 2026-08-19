# `brain-metric`

> Production metrics for classification, regression, detection, segmentation, NLP, ranking, clustering, and time series — with live tracking and report export.

## Overview

`brain-metric` provides a mathematically exact metrics suite over `brain-core` tensors and plain slices. It covers multi-class classification (accuracy, macro/micro/weighted PRF, ROC/PR AUC, calibration), regression (MSE/RMSE/MAE/MAPE/R² plus robust variants), detection mAP and IoU, segmentation mIoU, NLP (BLEU, METEOR-lite, perplexity, Levenshtein), ranking (MRR, NDCG@k), clustering purity, multilabel, imbalance, and time-series metrics — with a stateful `MetricTracker` and markdown/CSV report formatting.

## Features

- Classification: `accuracy_score`, `precision_recall_f1` (`AverageMode`: Macro/Micro/Weighted/None), `roc_auc_score`, `pr_auc_score`, `compute_calibration` (`CalibrationReport`)
- Regression: `mse_score`, `rmse_score`, `mae_score`, `r2_score`, `mape_score` plus robust `median_absolute_error`, `huber_metric`
- Detection & segmentation: `bbox_iou`, `mean_average_precision` (mAP), `miou_and_pixel_accuracy`
- NLP & ranking: `sentence_bleu`, `meteor_score_lite`, `perplexity_score`, `edit_distance_levenshtein`, `mean_reciprocal_rank`, `ndcg_at_k`
- Multilabel, imbalance, stats, time series: `exact_match_ratio`, `hamming_loss`, `matthews_correlation_coefficient`, `g_mean_score`, `pearson_correlation`, `mase_score`, `forecast_bias`
- Ops & utilities: `confusion_matrix`, `binarize_probs`, `threshold_sweep_roc`, `logits_to_predictions`, `stable_divide`, `topk_indices`, histogram binning
- `MetricTracker` rolling mean accumulator, `aggregate_metric_runs` (mean/variance/CI), `compare_models`, `format_markdown_report` / `format_csv_report`

## Modules

| Module | Description |
|---|---|
| `classification` | Accuracy, PRF, AUC (ROC/PR), calibration reports |
| `regression` | MSE/RMSE/MAE/MAPE/R² + robust (`robust.rs`) metrics |
| `detection` | `bbox_iou`, `mean_average_precision` (mAP) |
| `segmentation` | `miou_and_pixel_accuracy` |
| `nlp` | BLEU, METEOR-lite, perplexity, Levenshtein |
| `ranking` | MRR, NDCG@k |
| `cluster` | `cluster_purity` |
| `multilabel` | Exact-match ratio, Hamming loss |
| `imbalance` | MCC, geometric mean |
| `stats` | `pearson_correlation` |
| `time_series` | MASE, forecast bias |
| `ops` | Confusion matrix, threshold sweeps, binarization, argmax |
| `track` | `MetricTracker` epoch accumulator |
| `aggregate` | `aggregate_metric_runs` with variance and CI |
| `compare` | `compare_models` pairwise delta report |
| `report` | `format_markdown_report`, `format_csv_report` |
| `core` / `config` | `MetricKind`, `MetricValue`, `MetricError`, `MetricConfig`, `AverageMode` |
| `utils` | Stable divide, sorting, binning, top-k |

## Quick Start

```rust
use brain_metric::{accuracy_score, precision_recall_f1, AverageMode, MetricTracker};

let preds = vec![0, 1, 2, 1, 0];
let targets = vec![0, 1, 2, 0, 0];

let acc = accuracy_score(&preds, &targets);
let prf = precision_recall_f1(&preds, &targets, 3, AverageMode::Macro);
println!("acc={:.2} macro-f1={:.4}", acc, prf.f1);

let mut tracker = MetricTracker::new();
tracker.update("eval_loss", 0.35, 32);
tracker.update("eval_loss", 0.25, 32);
println!("mean eval loss: {:.4}", tracker.mean("eval_loss").unwrap());
```

## Testing

```bash
cargo test -p brain-metric -j 2
```

## Workspace Role

Depends only on `brain-core`. `brain-metric` is the evaluation layer of the framework: `brain-train` tracks training progress via `MetricTracker`, and benchmark/reporting workflows use its formatters to compare runs and models.