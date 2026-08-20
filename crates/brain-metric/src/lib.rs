//! # brain-metric
//!
//! Comprehensive, production-grade model evaluation metric library for the Brain Framework:
//! classification, regression, object detection, semantic segmentation, NLP, ranking,
//! clustering, time-series, calibration, tracking, cross-fold aggregation, and reporting.
//!
//! ## Architecture
//! - [`classification`] — Accuracy, Top-K, Balanced Accuracy, Precision/Recall/F1, ROC-AUC, PR-AUC, Calibration (ECE/MCE)
//! - [`regression`] — MSE, RMSE, MAE, MAPE, R², Explained Variance, Median AE, Huber Metric
//! - [`detection`] — Bounding box IoU (GIoU/DIoU/CIoU), COCO/VOC mAP (101-point / 11-point interpolation)
//! - [`segmentation`] — Mean IoU (mIoU), Pixel Accuracy, Dice coefficient, Boundary F1
//! - [`nlp`] — Sentence BLEU (1–4 with brevity penalty), ROUGE-1/2/L, METEOR-lite, Perplexity, Levenshtein distance
//! - [`ranking`] — Mean Reciprocal Rank (MRR), NDCG@k, MAP@k
//! - [`cluster`] — Cluster purity, Normalized Mutual Information (NMI), Adjusted Rand Index (ARI)
//! - [`time_series`] — Mean Absolute Scaled Error (MASE), Forecast Bias
//! - [`stats`] — Pearson correlation, Spearman rank correlation, Chi-Square goodness-of-fit
//! - [`multilabel`] — Exact match ratio, Hamming loss, subset accuracy
//! - [`imbalance`] — Matthews Correlation Coefficient (MCC), Geometric Mean (G-Mean)
//! - [`aggregate`] — Multi-run and cross-fold statistical aggregation (mean, std, 95% CI)
//! - [`track`] — `MetricTracker` accumulator for live training/evaluation loops
//! - [`compare`] — Pairwise model comparison and delta reporting
//! - [`report`] — Multi-format report export (Markdown, CSV, JSON, ASCII table)

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

pub mod aggregate;
pub mod classification;
pub mod cluster;
pub mod compare;
pub mod config;
pub mod core;
pub mod detection;
pub mod imbalance;
pub mod impl_;
pub mod multilabel;
pub mod nlp;
pub mod ops;
pub mod ranking;
pub mod regression;
pub mod report;
pub mod segmentation;
pub mod stats;
pub mod time_series;
pub mod track;
pub mod utils;

// ── Convenience re-exports ──────────────────────────────────────────────────
pub use aggregate::{aggregate_metric_runs, AggregateReport};
pub use classification::{
    accuracy_score, compute_calibration, pr_auc_score, precision_recall_f1, roc_auc_score,
    AucConfig, CalibrationReport, PrfReport,
};
pub use cluster::{cluster_purity, ClusterConfig};
pub use compare::{compare_models, CompareReport};
pub use config::{AverageMode, MetricConfig};
pub use core::{Metric, MetricError, MetricKind, MetricResult, MetricValue};
pub use detection::{bbox_iou, mean_average_precision, DetMetricConfig, MapConfig};
pub use imbalance::{g_mean_score, matthews_correlation_coefficient};
pub use impl_::{compute_metric, default_config, metric_names};
pub use multilabel::{exact_match_ratio, hamming_loss};
pub use nlp::{
    edit_distance_levenshtein, meteor_score_lite, perplexity_score, sentence_bleu, NlpMetricConfig,
};
pub use ops::{binarize_probs, confusion_matrix, logits_to_predictions, threshold_sweep_roc};
pub use ranking::{mean_reciprocal_rank, ndcg_at_k, RankingConfig};
pub use regression::{
    huber_metric, mae_score, mape_score, median_absolute_error, mse_score, r2_score, rmse_score,
    RobustMetricConfig,
};
pub use report::{format_csv_report, format_markdown_report, ReportFormat};
pub use segmentation::{miou_and_pixel_accuracy, SegMetricConfig};
pub use stats::{pearson_correlation, StatsConfig};
pub use time_series::{forecast_bias, mase_score, TsConfig};
pub use track::MetricTracker;
pub use utils::{bin_values_uniform, sort_descending_by_value, stable_divide, topk_indices};

/// Framework version string.
pub const VERSION: &str = "0.2.0";
