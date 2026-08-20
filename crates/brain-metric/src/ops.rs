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
    probs
        .iter()
        .map(|&p| if p >= threshold { 1 } else { 0 })
        .collect()
}

/// Performs a threshold sweep over binary probabilities to calculate True Positive and False Positive rates.
pub fn threshold_sweep_roc(
    probs: &[f64],
    targets: &[usize],
    num_thresholds: usize,
) -> (Vec<f64>, Vec<f64>) {
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
            if pred && actual {
                tp += 1;
            }
            if pred && !actual {
                fp += 1;
            }
        }

        let tpr = if total_pos > 0 {
            tp as f64 / total_pos as f64
        } else {
            0.0
        };
        let fpr = if total_neg > 0 {
            fp as f64 / total_neg as f64
        } else {
            0.0
        };
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
