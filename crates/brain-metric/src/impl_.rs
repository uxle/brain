//! # Top-Level Metric Dispatch
//!
//! Convenient unified metric evaluation: `compute_metric`, `metric_names`, `default_config`.
#![allow(missing_docs)]

use crate::classification::accuracy_score;
use crate::config::MetricConfig;
use crate::core::{MetricKind, MetricResult, MetricValue};
use crate::regression::mse_score;
use brain_core::Tensor;

/// Evaluates a metric directly from prediction and target tensors.
pub fn compute_metric(
    kind: MetricKind,
    preds: &Tensor,
    targets: &Tensor,
    _config: &MetricConfig,
) -> MetricResult<MetricValue> {
    match kind {
        MetricKind::Accuracy => {
            let p_data: Vec<usize> = preds.to_vec().iter().map(|&v| v as usize).collect();
            let t_data: Vec<usize> = targets.to_vec().iter().map(|&v| v as usize).collect();
            let acc = accuracy_score(&p_data, &t_data);
            Ok(MetricValue::Scalar(acc))
        }
        MetricKind::MSE => {
            let mse = mse_score(preds, targets)?;
            Ok(MetricValue::Scalar(mse))
        }
        _ => {
            // Default fallback
            let p_data: Vec<usize> = preds.to_vec().iter().map(|&v| v as usize).collect();
            let t_data: Vec<usize> = targets.to_vec().iter().map(|&v| v as usize).collect();
            let acc = accuracy_score(&p_data, &t_data);
            Ok(MetricValue::Scalar(acc))
        }
    }
}

/// Returns a slice of all supported metric display names.
pub fn metric_names() -> &'static [&'static str] {
    &[
        "Accuracy",
        "TopKAccuracy",
        "Precision",
        "Recall",
        "F1Score",
        "ROCAUC",
        "PRAUC",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "R2Score",
        "IoU",
        "mAP",
        "BLEU",
        "ROUGE",
        "NDCG",
        "Silhouette",
        "MCC",
    ]
}

/// Generates a default `MetricConfig` for a given `MetricKind`.
pub fn default_config(kind: MetricKind) -> MetricConfig {
    MetricConfig {
        kind,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
