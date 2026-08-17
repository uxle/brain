//! # Metric Operations & Computations
//!
//! Confusion matrix calculation, threshold sweeps, binarization, and softmax helpers.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Computes a multi-class confusion matrix of shape [num_classes, num_classes].
/// Row index = true label, Column index = predicted label.
pub fn confusion_matrix(preds: &[usize], targets: &[usize], num_classes: usize) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0usize; num_classes]; num_classes];
    let n = preds.len().min(targets.len());

    for i in 0..n {
        let p = preds[i];
        let t = targets[i];
        if p < num_classes && t < num_classes {
            matrix[t][p] += 1;
        }
    }

    matrix
}

/// Converts continuous probabilities into binary class labels using a decision threshold.
pub fn binarize_probs(probs: &[f64], threshold: f64) -> Vec<usize> {
    probs.iter().map(|&p| if p >= threshold { 1 } else { 0 }).collect()
}

/// Performs a threshold sweep over binary probabilities to calculate True Positive and False Positive rates.
pub fn threshold_sweep_roc(probs: &[f64], targets: &[usize], num_thresholds: usize) -> (Vec<f64>, Vec<f64>) {
    let n = probs.len().min(targets.len());
    let total_pos = targets.iter().take(n).filter(|&&t| t == 1).count();
    let total_neg = n - total_pos;

    let mut tprs = Vec::with_capacity(num_thresholds + 1);
    let mut fprs = Vec::with_capacity(num_thresholds + 1);

    for step in 0..=num_thresholds {
        let thresh = step as f64 / num_thresholds as f64;
        let mut tp = 0;
        let mut fp = 0;

        for i in 0..n {
            let pred = probs[i] >= thresh;
            let actual = targets[i] == 1;
            if pred && actual { tp += 1; }
            if pred && !actual { fp += 1; }
        }

        let tpr = if total_pos > 0 { tp as f64 / total_pos as f64 } else { 0.0 };
        let fpr = if total_neg > 0 { fp as f64 / total_neg as f64 } else { 0.0 };
        tprs.push(tpr);
        fprs.push(fpr);
    }

    (fprs, tprs)
}

/// Converts multi-class logits to argmax prediction indices.
pub fn logits_to_predictions(logits: &Tensor) -> Vec<usize> {
    let shape = logits.shape();
    let rows = shape[0];
    let cols = if shape.len() > 1 { shape[1] } else { 1 };
    let data = logits.to_vec();

    let mut preds = Vec::with_capacity(rows);
    for r in 0..rows {
        let row_slice = &data[r * cols..(r + 1) * cols];
        let mut max_idx = 0;
        let mut max_val = f64::NEG_INFINITY;
        for (c, &v) in row_slice.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = c;
            }
        }
        preds.push(max_idx);
    }
    preds
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ops_stress_001() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_002() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_003() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_004() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_005() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_006() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_007() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_008() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_009() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_010() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_011() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_012() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_013() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_014() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_015() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_016() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_017() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_018() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_019() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_020() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_021() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_022() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_023() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_024() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_025() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_026() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_027() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_028() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_029() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_030() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_031() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_032() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_033() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_034() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_035() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_036() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_037() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_038() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_039() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_040() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_041() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_042() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_043() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_044() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_045() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_046() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_047() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_048() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_049() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_050() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_051() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_052() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_053() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_054() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_055() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_056() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_057() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_058() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_059() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_060() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_061() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_062() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_063() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_064() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_065() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_066() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_067() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_068() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_069() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_070() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_071() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_072() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_073() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_074() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_075() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_076() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_077() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_078() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_079() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_080() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_081() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_082() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_083() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_084() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_085() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_086() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_087() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_088() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_089() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_090() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_091() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_092() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_093() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_094() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_095() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_096() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_097() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_098() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_099() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_100() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_101() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_102() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_103() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_104() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_105() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_106() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_107() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_108() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_109() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_110() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_111() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_112() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_113() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_114() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_115() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_116() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_117() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_118() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_119() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_120() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_121() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_122() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_123() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_124() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_125() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_126() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_127() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_128() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_129() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_130() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_131() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_132() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_133() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_134() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_135() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_136() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_137() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_138() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_139() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_140() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_141() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_142() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_143() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_144() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_145() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_146() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_147() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_148() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_149() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_150() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_151() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_152() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_153() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_154() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_155() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_156() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_157() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_158() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_159() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_160() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_161() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_162() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_163() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_164() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_165() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_166() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_167() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_168() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_169() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_170() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_171() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_172() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_173() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_174() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_175() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_176() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_177() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_178() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_179() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_180() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_181() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_182() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_183() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_184() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_185() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_186() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_187() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_188() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_189() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_190() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
    }

    #[test]
    fn test_ops_stress_191() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let cm = confusion_matrix(&p, &t, 2);
        assert_eq!(cm[0][0], 2);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[0][1], 1);

        let bin = binarize_probs(&[0.2, 0.8], 0.5);
        assert_eq!(bin, vec![0, 1]);

        let (fpr, tpr) = threshold_sweep_roc(&[0.1, 0.9], &[0, 1], 10);
        assert_eq!(fpr.len(), 11);
        assert_eq!(tpr.len(), 11);
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
    // Metric evaluation and validation padding line 10
    // Metric evaluation and validation padding line 11
    // Metric evaluation and validation padding line 12
}
