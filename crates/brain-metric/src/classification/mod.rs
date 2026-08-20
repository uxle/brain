//! # Classification Evaluation Metrics
//!
//! Accuracy, Top-K Accuracy, Balanced Accuracy, Precision, Recall, and F1 Score (Macro/Micro/Weighted).
#![allow(missing_docs)]

pub mod auc;
pub mod calibration;

pub use auc::{pr_auc_score, roc_auc_score, AucConfig};
pub use calibration::{compute_calibration, CalibrationReport};

use crate::config::AverageMode;
use crate::ops::confusion_matrix;
use crate::utils::stable_divide;

/// Computes standard multi-class classification accuracy (Top-1).
pub fn accuracy_score(preds: &[usize], targets: &[usize]) -> f64 {
    let n = preds.len().min(targets.len());
    if n == 0 {
        return 0.0;
    }
    let correct = preds
        .iter()
        .zip(targets.iter())
        .filter(|(&p, &t)| p == t)
        .count();
    correct as f64 / n as f64
}

/// Precision, Recall, and F1 score results per class or aggregated.
#[derive(Debug, Clone, Default)]
pub struct PrfReport {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub per_class_f1: Vec<f64>,
}

/// Computes Precision, Recall, and F1 score with specified averaging mode.
pub fn precision_recall_f1(
    preds: &[usize],
    targets: &[usize],
    num_classes: usize,
    average: AverageMode,
) -> PrfReport {
    let cm = confusion_matrix(preds, targets, num_classes);

    let mut precisions = vec![0.0f64; num_classes];
    let mut recalls = vec![0.0f64; num_classes];
    let mut f1s = vec![0.0f64; num_classes];
    let mut supports = vec![0usize; num_classes];

    for c in 0..num_classes {
        let tp = cm[c][c] as f64;
        let pred_pos: usize = (0..num_classes).map(|r| cm[r][c]).sum();
        let actual_pos: usize = cm[c].iter().sum();
        supports[c] = actual_pos;

        let p = stable_divide(tp, pred_pos as f64, 0.0);
        let r = stable_divide(tp, actual_pos as f64, 0.0);
        let f = stable_divide(2.0 * p * r, p + r, 0.0);

        precisions[c] = p;
        recalls[c] = r;
        f1s[c] = f;
    }

    match average {
        AverageMode::Macro => {
            let avg_p: f64 = precisions.iter().sum::<f64>() / num_classes as f64;
            let avg_r: f64 = recalls.iter().sum::<f64>() / num_classes as f64;
            let avg_f: f64 = f1s.iter().sum::<f64>() / num_classes as f64;
            PrfReport {
                precision: avg_p,
                recall: avg_r,
                f1: avg_f,
                per_class_f1: f1s,
            }
        }
        AverageMode::Weighted => {
            let total_support: usize = supports.iter().sum();
            let mut weighted_p = 0.0;
            let mut weighted_r = 0.0;
            let mut weighted_f = 0.0;
            if total_support > 0 {
                for c in 0..num_classes {
                    let w = supports[c] as f64 / total_support as f64;
                    weighted_p += precisions[c] * w;
                    weighted_r += recalls[c] * w;
                    weighted_f += f1s[c] * w;
                }
            }
            PrfReport {
                precision: weighted_p,
                recall: weighted_r,
                f1: weighted_f,
                per_class_f1: f1s,
            }
        }
        _ => {
            let acc = accuracy_score(preds, targets);
            PrfReport {
                precision: acc,
                recall: acc,
                f1: acc,
                per_class_f1: f1s,
            }
        }
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
