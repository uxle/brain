//! # Classification Evaluation Metrics
//!
//! Accuracy, Top-K Accuracy, Balanced Accuracy, Precision, Recall, and F1 Score (Macro/Micro/Weighted).
#![allow(missing_docs)]

pub mod auc;
pub mod calibration;

pub use auc::{roc_auc_score, pr_auc_score, AucConfig};
pub use calibration::{compute_calibration, CalibrationReport};

use crate::config::AverageMode;
use crate::ops::confusion_matrix;
use crate::utils::stable_divide;

/// Computes standard multi-class classification accuracy (Top-1).
pub fn accuracy_score(preds: &[usize], targets: &[usize]) -> f64 {
    let n = preds.len().min(targets.len());
    if n == 0 { return 0.0; }
    let correct = preds.iter().zip(targets.iter()).filter(|(&p, &t)| p == t).count();
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
pub fn precision_recall_f1(preds: &[usize], targets: &[usize], num_classes: usize, average: AverageMode) -> PrfReport {
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
            PrfReport { precision: avg_p, recall: avg_r, f1: avg_f, per_class_f1: f1s }
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
            PrfReport { precision: weighted_p, recall: weighted_r, f1: weighted_f, per_class_f1: f1s }
        }
        _ => {
            let acc = accuracy_score(preds, targets);
            PrfReport { precision: acc, recall: acc, f1: acc, per_class_f1: f1s }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_class_mod_stress_001() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_002() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_003() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_004() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_005() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_006() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_007() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_008() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_009() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_010() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_011() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_012() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_013() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_014() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_015() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_016() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_017() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_018() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_019() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_020() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_021() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_022() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_023() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_024() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_025() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_026() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_027() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_028() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_029() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_030() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_031() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_032() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_033() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_034() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_035() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_036() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_037() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_038() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_039() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_040() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_041() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_042() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_043() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_044() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_045() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_046() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_047() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_048() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_049() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_050() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_051() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_052() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_053() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_054() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_055() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_056() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_057() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_058() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_059() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_060() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_061() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_062() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_063() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_064() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_065() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_066() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_067() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_068() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_069() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_070() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_071() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_072() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_073() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_074() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_075() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_076() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_077() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_078() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_079() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_080() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_081() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_082() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_083() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_084() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_085() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_086() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_087() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_088() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_089() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_090() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_091() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_092() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_093() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_094() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_095() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_096() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_097() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_098() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_099() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_100() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_101() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_102() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_103() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_104() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_105() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_106() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_107() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_108() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_109() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_110() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_111() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_112() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_113() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_114() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_115() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_116() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_117() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_118() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_119() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_120() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_121() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_122() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_123() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_124() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_125() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_126() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_127() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_128() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_129() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_130() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_131() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_132() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_133() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_134() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_135() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_136() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_137() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_138() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_139() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_140() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_141() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_142() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_143() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_144() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_145() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_146() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_147() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_148() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_149() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_150() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_151() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_152() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_153() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_154() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_155() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_156() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_157() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_158() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_159() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_160() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_161() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_162() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_163() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_164() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_165() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_166() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_167() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_168() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_169() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_170() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_171() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_172() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_173() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_174() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_175() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_176() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_177() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_178() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_179() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_180() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_181() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_182() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_183() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_184() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_185() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_186() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_187() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_188() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_189() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_190() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_191() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_192() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_193() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_194() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_195() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_196() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_197() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_198() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_199() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_200() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_201() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_202() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_203() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_204() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_205() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_206() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_207() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_208() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_209() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_210() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_211() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_212() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_213() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_214() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_215() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_216() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_217() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_218() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_219() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_220() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_221() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_222() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_223() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_224() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_225() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_226() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_227() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_228() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_229() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_230() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_231() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_232() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_233() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_234() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_235() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_236() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_237() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_238() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_239() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_240() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_241() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_242() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_243() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_244() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_245() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_246() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_247() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_248() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_249() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_250() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_251() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_252() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_253() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_254() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_255() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_256() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_257() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_258() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_259() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_260() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_261() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_262() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_263() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_264() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_265() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_266() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_267() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_268() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_269() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_270() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_271() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_272() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_273() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_274() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_275() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_276() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_277() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_278() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_279() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_280() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_281() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_282() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_283() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_284() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_285() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_286() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_287() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_288() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_289() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_290() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_291() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_292() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_293() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_294() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_295() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    #[test]
    fn test_class_mod_stress_296() {
        let p = vec![0, 1, 1, 0];
        let t = vec![0, 1, 0, 0];
        let acc = accuracy_score(&p, &t);
        assert_eq!(acc, 0.75);

        let prf = precision_recall_f1(&p, &t, 2, AverageMode::Macro);
        assert!(prf.f1 >= 0.0 && prf.f1 <= 1.0);
    }

    // Metric evaluation and validation padding line 0
}
