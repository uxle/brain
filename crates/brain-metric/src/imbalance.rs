//! # Imbalanced Classification Metrics
//!
//! Geometric Mean (G-Mean), Matthews Correlation Coefficient (MCC), Informedness, and Markedness.
#![allow(missing_docs)]

use crate::utils::stable_divide;

/// Matthews Correlation Coefficient (MCC) from binary confusion components: (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
pub fn matthews_correlation_coefficient(tp: usize, tn: usize, fp: usize, fn_: usize) -> f64 {
    let num = (tp as f64 * tn as f64) - (fp as f64 * fn_ as f64);
    let den = ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();
    stable_divide(num, den, 0.0)
}

/// Geometric Mean (G-Mean) = sqrt(Sensitivity * Specificity).
pub fn g_mean_score(tp: usize, tn: usize, fp: usize, fn_: usize) -> f64 {
    let sensitivity = stable_divide(tp as f64, (tp + fn_) as f64, 0.0);
    let specificity = stable_divide(tn as f64, (tn + fp) as f64, 0.0);
    (sensitivity * specificity).sqrt()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
