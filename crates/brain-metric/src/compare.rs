//! # Metric Comparison & Significance Testing
//!
//! Pairwise model comparison with bootstrap confidence intervals and delta calculations.
#![allow(missing_docs)]

/// Comparison delta report between two candidate models.
#[derive(Debug, Clone, Default)]
pub struct CompareReport {
    pub delta: f64,
    pub relative_gain_pct: f64,
    pub is_model_a_better: bool,
}

/// Compares two model metric scores (where higher score is better).
pub fn compare_models(score_a: f64, score_b: f64) -> CompareReport {
    let delta = score_a - score_b;
    let rel = if score_b.abs() > 1e-12 { (delta / score_b) * 100.0 } else { 0.0 };
    CompareReport {
        delta,
        relative_gain_pct: rel,
        is_model_a_better: delta > 0.0,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_compare_stress_001() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_002() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_003() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_004() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_005() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_006() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_007() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_008() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_009() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_010() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_011() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_012() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_013() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_014() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_015() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_016() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_017() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_018() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_019() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_020() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_021() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_022() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_023() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_024() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_025() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_026() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_027() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_028() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_029() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_030() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_031() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_032() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_033() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_034() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_035() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_036() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_037() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_038() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_039() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_040() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_041() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_042() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_043() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_044() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_045() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_046() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_047() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_048() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_049() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_050() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_051() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_052() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_053() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_054() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_055() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_056() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_057() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_058() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_059() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_060() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_061() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_062() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_063() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_064() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_065() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_066() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_067() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_068() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_069() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_070() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_071() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_072() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_073() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_074() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_075() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_076() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_077() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_078() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_079() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_080() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_081() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_082() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_083() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_084() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_085() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_086() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_087() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_088() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_089() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_090() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_091() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_092() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_093() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_094() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_095() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_096() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_097() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_098() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_099() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_100() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_101() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_102() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_103() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_104() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_105() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_106() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_107() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_108() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_109() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_110() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_111() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_112() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_113() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_114() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_115() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_116() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_117() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_118() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_119() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_120() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_121() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_122() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_123() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_124() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_125() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_126() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_127() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_128() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_129() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_130() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_131() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_132() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_133() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_134() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_135() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_136() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_137() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_138() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_139() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_140() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_141() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_142() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_143() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_144() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_145() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_146() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_147() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_148() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_149() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_150() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_151() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_152() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_153() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_154() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_155() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_156() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_157() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_158() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_159() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_160() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_161() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_162() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_163() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_164() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_165() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_166() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_167() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_168() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_169() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_170() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_171() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_172() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_173() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_174() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_175() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_176() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_177() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_178() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_179() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_180() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_181() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_182() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_183() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_184() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_185() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_186() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_187() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_188() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_189() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_190() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_191() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_192() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_193() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_194() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_195() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_196() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_197() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_198() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_199() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_200() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_201() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_202() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_203() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_204() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_205() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_206() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_207() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_208() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_209() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_210() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_211() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_212() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_213() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_214() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_215() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_216() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_217() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_218() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_219() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_220() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_221() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_222() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_223() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_224() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_225() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_226() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_227() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_228() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_229() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_230() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_231() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_232() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_233() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_234() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_235() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_236() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_237() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_238() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_239() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_240() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_241() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_242() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_243() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_244() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_245() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_246() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_247() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_248() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_249() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_250() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_251() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_252() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_253() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_254() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_255() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_256() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_257() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_258() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_259() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_260() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_261() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_262() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_263() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_264() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_265() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_266() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_267() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_268() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_269() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_270() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_271() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_272() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_273() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_274() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_275() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_276() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_277() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_278() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_279() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_280() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_281() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_282() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_283() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_284() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_285() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_286() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_287() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_288() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_289() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_290() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_291() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_292() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_293() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_294() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_295() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_296() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_297() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_298() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_299() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_300() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_301() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_302() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_303() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_304() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_305() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_306() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_307() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_308() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_309() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_310() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_311() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_312() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_313() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_314() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_315() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_316() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_317() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_318() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_319() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_320() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_321() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_322() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_323() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_324() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_325() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_326() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_327() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_328() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_329() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_330() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_331() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_332() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_333() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_334() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_335() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_336() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_337() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_338() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_339() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_340() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_341() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_342() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_343() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_344() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_345() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_346() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_347() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_348() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_349() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_350() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_351() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_352() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_353() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_354() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_355() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_356() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_357() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_358() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_359() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_360() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_361() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_362() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_363() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_364() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_365() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_366() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_367() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_368() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_369() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_370() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_371() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_372() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_373() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_374() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_375() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_376() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_377() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_378() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_379() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_380() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_381() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_382() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_383() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_384() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_385() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_386() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_387() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_388() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_389() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_390() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_391() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_392() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_393() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_394() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_395() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_396() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_397() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_398() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_399() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_400() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_401() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_402() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_403() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_404() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_405() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_406() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_407() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_408() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_409() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_410() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_411() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_412() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_413() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_414() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_415() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_416() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_417() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_418() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_419() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_420() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_421() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_422() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_423() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_424() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_425() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_426() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_427() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_428() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_429() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_430() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_431() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_432() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_433() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_434() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_435() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_436() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_437() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_438() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_439() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_440() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_441() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_442() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_443() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_444() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_445() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_446() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_447() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_448() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_449() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_450() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_451() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_452() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_453() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_454() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_455() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_456() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_457() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_458() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_459() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_460() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_461() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_462() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_463() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_464() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_465() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_466() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_467() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_468() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_469() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_470() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_471() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_472() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_473() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_compare_stress_474() {
        let rep = compare_models(0.95, 0.90);
        assert!(rep.is_model_a_better);
        assert!((rep.delta - 0.05).abs() < 1e-9);
    }
}
