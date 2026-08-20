//! # Area Under the Curve (ROC-AUC & PR-AUC)
//!
//! Sorting-based trapezoidal numerical integration for ROC-AUC, PR-AUC, and Multi-Class One-vs-Rest AUC.
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
        let precision = tp as f64 / (tp + fp) as f64;

        auc += (recall - prev_recall) * precision;
        prev_recall = recall;
    }

    auc
}

/// Computes multi-class One-vs-Rest (OvR) Macro-Averaged ROC-AUC.
pub fn multiclass_roc_auc(probs: &[Vec<f64>], targets: &[usize], num_classes: usize) -> f64 {
    if num_classes <= 1 || probs.is_empty() {
        return 1.0;
    }

    let mut sum_auc = 0.0;
    let mut valid_classes = 0;

    for c in 0..num_classes {
        let class_probs: Vec<f64> = probs
            .iter()
            .map(|p| p.get(c).copied().unwrap_or(0.0))
            .collect();
        let class_targets: Vec<usize> = targets
            .iter()
            .map(|&t| if t == c { 1 } else { 0 })
            .collect();

        let num_pos = class_targets.iter().filter(|&&t| t == 1).count();
        if num_pos > 0 && num_pos < targets.len() {
            sum_auc += roc_auc_score(&class_probs, &class_targets);
            valid_classes += 1;
        }
    }

    stable_divide(sum_auc, valid_classes as f64, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roc_auc_perfect_and_imperfect() {
        // Perfect ranking
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        assert!((roc_auc_score(&probs, &targets) - 1.0).abs() < 1e-6);

        // Inverted ranking
        let inv_probs = vec![0.1, 0.2, 0.8, 0.9];
        assert!((roc_auc_score(&inv_probs, &targets) - 0.0).abs() < 1e-6);
    }
}
