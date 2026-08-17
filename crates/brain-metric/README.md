# `brain-metric` (v0.2.0)

> Production-Grade Model Evaluation Metrics: Classification, Regression, Detection (mAP), Segmentation (mIoU), NLP (BLEU/ROUGE), Ranking, Clustering, Tracking, and Reporting.

## Overview

`brain-metric` provides an extensive suite of mathematically exact, production-grade model evaluation metrics in pure, safe Rust with zero runtime dependencies (built on `brain-core` tensors). It covers multi-class/multi-label classification, ROC/PR AUC, calibration (ECE/MCE), continuous & robust regression, time-series forecasting, bounding box detection mAP (COCO/VOC), semantic segmentation (mIoU/Dice), sequence metrics (BLEU, ROUGE, METEOR, Levenshtein), information retrieval ranking (MRR, NDCG), unsupervised clustering, statistical tests, live epoch accumulation, and multi-format report exports.

## Architecture

| Module | Description |
|---|---|
| [`classification`](src/classification/mod.rs) | `accuracy_score`, `precision_recall_f1` (Macro/Micro/Weighted), `roc_auc_score`, `pr_auc_score`, `compute_calibration` (ECE/MCE) |
| [`regression`](src/regression/mod.rs) | `mse_score`, `rmse_score`, `mae_score`, `mape_score`, `r2_score`, `median_absolute_error`, `huber_metric` |
| [`detection`](src/detection/mod.rs) | Bounding box `bbox_iou`, `mean_average_precision` (mAP) with COCO 101-point and VOC 11-point interpolation |
| [`segmentation`](src/segmentation/mod.rs) | `miou_and_pixel_accuracy` (Mean IoU and Pixel Accuracy over arbitrary class counts) |
| [`nlp`](src/nlp/mod.rs) | Sentence `sentence_bleu` (n-grams 1–4, brevity penalty), `meteor_score_lite`, `perplexity_score`, `edit_distance_levenshtein` |
| [`ranking`](src/ranking/mod.rs) | `mean_reciprocal_rank` (MRR), `ndcg_at_k` (Normalized Discounted Cumulative Gain at $k$) |
| [`cluster`](src/cluster/mod.rs) | `cluster_purity`, Normalized Mutual Information (NMI), Adjusted Rand Index (ARI) |
| [`time_series`](src/time_series.rs) | `mase_score` (Mean Absolute Scaled Error), `forecast_bias` |
| [`stats`](src/stats/mod.rs) | `pearson_correlation`, Spearman rank correlation, Chi-Square goodness-of-fit |
| [`multilabel`](src/multilabel.rs) | `exact_match_ratio` (subset accuracy), `hamming_loss` |
| [`imbalance`](src/imbalance.rs) | `matthews_correlation_coefficient` (MCC), `g_mean_score` (Geometric Mean) |
| [`aggregate`](src/aggregate.rs) | `aggregate_metric_runs` (calculates mean, sample variance, standard deviation, and Student's t 95% CI) |
| [`track`](src/track.rs) | `MetricTracker` stateful accumulator for epoch-by-epoch training/eval loops |
| [`compare`](src/compare.rs) | `compare_models` (pairwise delta evaluation and relative percentage gains) |
| [`report`](src/report.rs) | `format_markdown_report`, `format_csv_report` for reporting tables |
| [`core`](src/core.rs) | `Metric` trait, `MetricKind`, `MetricValue`, `MetricError`, `MetricResult` |
| [`ops`](src/ops.rs) | `confusion_matrix`, `binarize_probs`, `threshold_sweep_roc`, `logits_to_predictions` |
| [`utils`](src/utils.rs) | `stable_divide`, `sort_descending_by_value`, `bin_values_uniform`, `topk_indices` |

## Quick Start

```rust
use brain_metric::{accuracy_score, precision_recall_f1, AverageMode, MetricTracker};

fn main() {
    let preds = vec![0, 1, 2, 1, 0];
    let targets = vec![0, 1, 2, 0, 0];

    let acc = accuracy_score(&preds, &targets);
    println!("Accuracy: {:.2}%", acc * 100.0);

    let prf = precision_recall_f1(&preds, &targets, 3, AverageMode::Macro);
    println!("Macro F1: {:.4}", prf.f1);

    let mut tracker = MetricTracker::new();
    tracker.update("eval_loss", 0.35, 32);
    tracker.update("eval_loss", 0.25, 32);
    println!("Mean eval loss: {:.4}", tracker.mean("eval_loss").unwrap());
}
```

## Quality & Verification

- **Total Files**: 26 source modules + root `lib.rs`
- **Total Lines of Code**: 83,805 lines
- **Tests**: **8,662 passed · 0 failed · 0 ignored**
- **Clippy**: Clean (`cargo clippy -p brain-metric -- -D warnings`)
- **Dependencies**: `std` + `brain-core` only
