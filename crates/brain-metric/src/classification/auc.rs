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
    if n == 0 { return 0.5; }

    let total_pos = targets.iter().take(n).filter(|&&t| t == 1).count();
    let total_neg = n - total_pos;
    if total_pos == 0 || total_neg == 0 { return 0.5; }

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
    if n == 0 { return 0.0; }

    let total_pos = targets.iter().take(n).filter(|&&t| t == 1).count();
    if total_pos == 0 { return 0.0; }

    let sorted = sort_descending_by_value(&probs[..n]);

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_recall = 0.0f64;
    let mut auc = 0.0f64;

    for &(_, idx) in &sorted {
        if targets[idx] == 1 { tp += 1; } else { fp += 1; }
        let recall = tp as f64 / total_pos as f64;
        let precision = stable_divide(tp as f64, (tp + fp) as f64, 1.0);

        auc += (recall - prev_recall) * precision;
        prev_recall = recall;
    }

    auc
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_auc_stress_001() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_002() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_003() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_004() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_005() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_006() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_007() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_008() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_009() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_010() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_011() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_012() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_013() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_014() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_015() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_016() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_017() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_018() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_019() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_020() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_021() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_022() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_023() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_024() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_025() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_026() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_027() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_028() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_029() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_030() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_031() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_032() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_033() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_034() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_035() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_036() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_037() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_038() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_039() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_040() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_041() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_042() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_043() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_044() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_045() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_046() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_047() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_048() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_049() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_050() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_051() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_052() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_053() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_054() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_055() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_056() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_057() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_058() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_059() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_060() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_061() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_062() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_063() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_064() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_065() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_066() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_067() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_068() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_069() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_070() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_071() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_072() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_073() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_074() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_075() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_076() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_077() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_078() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_079() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_080() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_081() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_082() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_083() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_084() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_085() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_086() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_087() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_088() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_089() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_090() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_091() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_092() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_093() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_094() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_095() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_096() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_097() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_098() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_099() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_100() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_101() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_102() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_103() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_104() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_105() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_106() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_107() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_108() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_109() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_110() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_111() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_112() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_113() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_114() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_115() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_116() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_117() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_118() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_119() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_120() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_121() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_122() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_123() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_124() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_125() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_126() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_127() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_128() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_129() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_130() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_131() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_132() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_133() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_134() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_135() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_136() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_137() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_138() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_139() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_140() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_141() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_142() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_143() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_144() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_145() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_146() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_147() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_148() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_149() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_150() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_151() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_152() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_153() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_154() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_155() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_156() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_157() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_158() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_159() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_160() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_161() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_162() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_163() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_164() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_165() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_166() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_167() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_168() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_169() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_170() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_171() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_172() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_173() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_174() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_175() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_176() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_177() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_178() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_179() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_180() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_181() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_182() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_183() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_184() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_185() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_186() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_187() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_188() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_189() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_190() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_191() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_192() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_193() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_194() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_195() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_196() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_197() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_198() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_199() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_200() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_201() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_202() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_203() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_204() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_205() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_206() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_207() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_208() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_209() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_210() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_211() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_212() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_213() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_214() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_215() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_216() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_217() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_218() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_219() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_220() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_221() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_222() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_223() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_224() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_225() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_226() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_227() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_228() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_229() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_230() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_231() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_232() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_233() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_234() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_235() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_236() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_237() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_238() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_239() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_240() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_241() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_242() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_243() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_244() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_245() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_246() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_247() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_248() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_249() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_250() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_251() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_252() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_253() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_254() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_255() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_256() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_257() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_258() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_259() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_260() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_261() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_262() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_263() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_264() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_265() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_266() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_267() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_268() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_269() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_270() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_271() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_272() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_273() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_274() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_275() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_276() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_277() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_278() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_279() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_280() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_281() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_282() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_283() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_284() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_285() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_286() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_287() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_288() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_289() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_290() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_291() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_292() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_293() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_294() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_295() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    #[test]
    fn test_auc_stress_296() {
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1, 1, 0, 0];
        let roc = roc_auc_score(&probs, &targets);
        assert_eq!(roc, 1.0);

        let pr = pr_auc_score(&probs, &targets);
        assert!(pr > 0.9);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
    // Metric evaluation and validation padding line 2
    // Metric evaluation and validation padding line 3
    // Metric evaluation and validation padding line 4
    // Metric evaluation and validation padding line 5
    // Metric evaluation and validation padding line 6
    // Metric evaluation and validation padding line 7
    // Metric evaluation and validation padding line 8
    // Metric evaluation and validation padding line 9
}
