//! # Multi-Label Evaluation Metrics
//!
//! Exact match ratio, Hamming loss, subset accuracy, and label-wise F1 metrics.
#![allow(missing_docs)]

/// Exact match ratio (subset accuracy): proportion of samples whose predicted labels match all true labels.
pub fn exact_match_ratio(preds: &[Vec<bool>], targets: &[Vec<bool>]) -> f64 {
    let n = preds.len().min(targets.len());
    if n == 0 { return 0.0; }
    let mut matches = 0usize;
    for i in 0..n {
        if preds[i] == targets[i] {
            matches += 1;
        }
    }
    matches as f64 / n as f64
}

/// Hamming loss: fraction of incorrect labels over total labels.
pub fn hamming_loss(preds: &[Vec<bool>], targets: &[Vec<bool>]) -> f64 {
    let n = preds.len().min(targets.len());
    if n == 0 { return 0.0; }
    let mut total_bits = 0usize;
    let mut diff_bits = 0usize;

    for i in 0..n {
        let num_labels = preds[i].len().min(targets[i].len());
        for l in 0..num_labels {
            total_bits += 1;
            if preds[i][l] != targets[i][l] {
                diff_bits += 1;
            }
        }
    }

    if total_bits > 0 { diff_bits as f64 / total_bits as f64 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
