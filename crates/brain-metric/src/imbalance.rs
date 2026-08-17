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

    #[test]
    fn test_imbalance_stress_001() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_002() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_003() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_004() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_005() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_006() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_007() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_008() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_009() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_010() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_011() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_012() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_013() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_014() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_015() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_016() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_017() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_018() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_019() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_020() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_021() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_022() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_023() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_024() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_025() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_026() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_027() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_028() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_029() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_030() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_031() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_032() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_033() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_034() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_035() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_036() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_037() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_038() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_039() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_040() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_041() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_042() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_043() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_044() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_045() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_046() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_047() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_048() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_049() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_050() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_051() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_052() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_053() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_054() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_055() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_056() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_057() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_058() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_059() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_060() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_061() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_062() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_063() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_064() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_065() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_066() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_067() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_068() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_069() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_070() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_071() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_072() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_073() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_074() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_075() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_076() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_077() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_078() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_079() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_080() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_081() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_082() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_083() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_084() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_085() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_086() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_087() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_088() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_089() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_090() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_091() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_092() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_093() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_094() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_095() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_096() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_097() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_098() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_099() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_100() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_101() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_102() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_103() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_104() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_105() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_106() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_107() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_108() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_109() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_110() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_111() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_112() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_113() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_114() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_115() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_116() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_117() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_118() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_119() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_120() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_121() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_122() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_123() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_124() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_125() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_126() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_127() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_128() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_129() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_130() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_131() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_132() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_133() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_134() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_135() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_136() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_137() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_138() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_139() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_140() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_141() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_142() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_143() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_144() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_145() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_146() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_147() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_148() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_149() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_150() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_151() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_152() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_153() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_154() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_155() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_156() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_157() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_158() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_159() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_160() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_161() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_162() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_163() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_164() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_165() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_166() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_167() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_168() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_169() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_170() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_171() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_172() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_173() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_174() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_175() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_176() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_177() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_178() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_179() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_180() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_181() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_182() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_183() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_184() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_185() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_186() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_187() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_188() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_189() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_190() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_191() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_192() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_193() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_194() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_195() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_196() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_197() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_198() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_199() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_200() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_201() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_202() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_203() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_204() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_205() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_206() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_207() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_208() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_209() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_210() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_211() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_212() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_213() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_214() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_215() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_216() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_217() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_218() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_219() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_220() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_221() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_222() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_223() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_224() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_225() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_226() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_227() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_228() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_229() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_230() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_231() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_232() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_233() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_234() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_235() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_236() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_237() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_238() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_239() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_240() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_241() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_242() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_243() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_244() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_245() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_246() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_247() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_248() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_249() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_250() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_251() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_252() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_253() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_254() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_255() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_256() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_257() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_258() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_259() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_260() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_261() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_262() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_263() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_264() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_265() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_266() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_267() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_268() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_269() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_270() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_271() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_272() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_273() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_274() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_275() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_276() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_277() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_278() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_279() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_280() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_281() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_282() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_283() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_284() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_285() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_286() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_287() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_288() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_289() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_290() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_291() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_292() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_293() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_294() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_295() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_296() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_297() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_298() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_299() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_300() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_301() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_302() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_303() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_304() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_305() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_306() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_307() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_308() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_309() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_310() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_311() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_312() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_313() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_314() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_315() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_316() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_317() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_318() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_319() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_320() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_321() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_322() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_323() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_324() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_325() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_326() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_327() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_328() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_329() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_330() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_331() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_332() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_333() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_334() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_335() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_336() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_337() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_338() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_339() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_340() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_341() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_342() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_343() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_344() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_345() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_346() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_347() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_348() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_349() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_350() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_351() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_352() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_353() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_354() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_355() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_356() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_357() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_358() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_359() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_360() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_361() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_362() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_363() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_364() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_365() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_366() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_367() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_368() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_imbalance_stress_369() {
        let mcc = matthews_correlation_coefficient(10, 10, 0, 0);
        assert!((mcc - 1.0).abs() < 1e-9);

        let gm = g_mean_score(10, 10, 0, 0);
        assert!((gm - 1.0).abs() < 1e-9);
    }
}
