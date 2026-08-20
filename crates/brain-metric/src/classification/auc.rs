//! # Area Under the Curve (ROC-AUC & PR-AUC)
//!
//! Sorting-based trapezoidal numerical integration for ROC-AUC and Precision-Recall AUC.
#![allow(missing_docs)]

use crate::utils::{sort_descending_by_value, stable_divide};

/// Configuration for AUC integration.
#[derive(Debug, Clone, Default)]
pub struct AucConfig {
    pub one_vs_rest: bool,
}

/// Computes binary ROC-AUC using trapezoidal numerical integration over sorted threshold pairs.
pub fn roc_auc_score(probs: &[f64], targets: &[usize]) -> f64 {
    let n = probs.len().min(targets.len());
    if n == 0 {
        return 0.5;
    }

    let total_pos = targets.iter().take(n).filter(|&&t| t == 1).count();
    let total_neg = n - total_pos;
    if total_pos == 0 || total_neg == 0 {
        return 0.5;
    }

    let sorted = sort_descending_by_value(&probs[..n]);

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_fpr = 0.0f64;
    let mut prev_tpr = 0.0f64;
    let mut auc = 0.0f64;

    for &(_, idx) in &sorted {
        if targets[idx] == 1 {
            tp += 1;
        } else {
            fp += 1;
        }

        let tpr = tp as f64 / total_pos as f64;
        let fpr = fp as f64 / total_neg as f64;

        // Trapezoidal integration step: (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    auc
}

/// Computes Precision-Recall Area Under the Curve (PR-AUC).
pub fn pr_auc_score(probs: &[f64], targets: &[usize]) -> f64 {
    let n = probs.len().min(targets.len());
    if n == 0 {
        return 0.0;
    }

    let total_pos = targets.iter().take(n).filter(|&&t| t == 1).count();
    if total_pos == 0 {
        return 0.0;
    }

    let sorted = sort_descending_by_value(&probs[..n]);

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_recall = 0.0f64;
    let mut auc = 0.0f64;

    for &(_, idx) in &sorted {
        if targets[idx] == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        let recall = tp as f64 / total_pos as f64;
        let precision = stable_divide(tp as f64, (tp + fp) as f64, 1.0);

        auc += (recall - prev_recall) * precision;
        prev_recall = recall;
    }

    auc
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
