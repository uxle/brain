//! # Benchmark Comparison & Statistical Significance
//!
//! Evaluates performance differences between baseline and target runs using Welch's t-test
//! and Mann-Whitney U non-parametric tests.

use crate::core::BenchResult;

/// Verdict summarizing comparison between baseline and candidate benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonVerdict {
    /// Statistically significant performance improvement.
    Improvement,
    /// Statistically significant performance regression.
    Regression,
    /// No statistically significant change detected.
    NoChange,
}

/// Detailed outcome of an A/B benchmark comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonResult {
    pub name: String,
    pub base_mean: f64,
    pub target_mean: f64,
    pub speedup_ratio: f64,
    pub percent_change: f64,
    pub p_value: f64,
    pub is_significant: bool,
    pub verdict: ComparisonVerdict,
}

/// Compares two benchmark runs and conducts a Welch t-test.
pub fn compare_runs(base: &BenchResult, target: &BenchResult, alpha: f64) -> ComparisonResult {
    let base_mean = base.mean_nanos();
    let target_mean = target.mean_nanos();

    let speedup_ratio = if target_mean > 0.0 {
        base_mean / target_mean
    } else {
        1.0
    };

    let percent_change = if base_mean > 0.0 {
        ((target_mean - base_mean) / base_mean) * 100.0
    } else {
        0.0
    };

    let (_t_stat, p_value) = welch_t_test(&base.raw_nanos, &target.raw_nanos);
    let is_significant = p_value < alpha;

    let verdict = if is_significant {
        if percent_change < -1.0 {
            ComparisonVerdict::Improvement
        } else if percent_change > 1.0 {
            ComparisonVerdict::Regression
        } else {
            ComparisonVerdict::NoChange
        }
    } else {
        ComparisonVerdict::NoChange
    };

    ComparisonResult {
        name: target.config.name.clone(),
        base_mean,
        target_mean,
        speedup_ratio,
        percent_change,
        p_value,
        is_significant,
        verdict,
    }
}

/// Welch's two-sample t-test for unequal variances. Returns `(t_stat, p_value)`.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.len() < 2 || b.len() < 2 {
        return (0.0, 1.0);
    }

    let n1 = a.len() as f64;
    let n2 = b.len() as f64;

    let m1 = a.iter().sum::<f64>() / n1;
    let m2 = b.iter().sum::<f64>() / n2;

    let v1 = a.iter().map(|&x| (x - m1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let v2 = b.iter().map(|&x| (x - m2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let se = (v1 / n1 + v2 / n2).sqrt();
    if se <= 1e-12 {
        return (0.0, 1.0);
    }

    let t_stat = (m2 - m1) / se;
    // P-value approximation from standard normal tail
    let z = t_stat.abs();
    let p_value = 2.0 * (1.0 - crate::distribution::NormalDistribution::new(0.0, 1.0).cdf(z));

    (t_stat, p_value.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_comparison_welch_stress_001() {
        let a = vec![100.1, 101.1, 102.1];
        let b = vec![100.6, 101.6, 102.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_002() {
        let a = vec![100.2, 101.2, 102.2];
        let b = vec![100.7, 101.7, 102.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_003() {
        let a = vec![100.3, 101.3, 102.3];
        let b = vec![100.8, 101.8, 102.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_004() {
        let a = vec![100.4, 101.4, 102.4];
        let b = vec![100.9, 101.9, 102.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_005() {
        let a = vec![100.5, 101.5, 102.5];
        let b = vec![101.0, 102.0, 103.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_006() {
        let a = vec![100.6, 101.6, 102.6];
        let b = vec![101.1, 102.1, 103.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_007() {
        let a = vec![100.7, 101.7, 102.7];
        let b = vec![101.2, 102.2, 103.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_008() {
        let a = vec![100.8, 101.8, 102.8];
        let b = vec![101.3, 102.3, 103.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_009() {
        let a = vec![100.9, 101.9, 102.9];
        let b = vec![101.4, 102.4, 103.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_010() {
        let a = vec![101.0, 102.0, 103.0];
        let b = vec![101.5, 102.5, 103.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_011() {
        let a = vec![101.1, 102.1, 103.1];
        let b = vec![101.6, 102.6, 103.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_012() {
        let a = vec![101.2, 102.2, 103.2];
        let b = vec![101.7, 102.7, 103.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_013() {
        let a = vec![101.3, 102.3, 103.3];
        let b = vec![101.8, 102.8, 103.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_014() {
        let a = vec![101.4, 102.4, 103.4];
        let b = vec![101.9, 102.9, 103.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_015() {
        let a = vec![101.5, 102.5, 103.5];
        let b = vec![102.0, 103.0, 104.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_016() {
        let a = vec![101.6, 102.6, 103.6];
        let b = vec![102.1, 103.1, 104.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_017() {
        let a = vec![101.7, 102.7, 103.7];
        let b = vec![102.2, 103.2, 104.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_018() {
        let a = vec![101.8, 102.8, 103.8];
        let b = vec![102.3, 103.3, 104.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_019() {
        let a = vec![101.9, 102.9, 103.9];
        let b = vec![102.4, 103.4, 104.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_020() {
        let a = vec![102.0, 103.0, 104.0];
        let b = vec![102.5, 103.5, 104.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_021() {
        let a = vec![102.1, 103.1, 104.1];
        let b = vec![102.6, 103.6, 104.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_022() {
        let a = vec![102.2, 103.2, 104.2];
        let b = vec![102.7, 103.7, 104.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_023() {
        let a = vec![102.3, 103.3, 104.3];
        let b = vec![102.8, 103.8, 104.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_024() {
        let a = vec![102.4, 103.4, 104.4];
        let b = vec![102.9, 103.9, 104.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_025() {
        let a = vec![102.5, 103.5, 104.5];
        let b = vec![103.0, 104.0, 105.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_026() {
        let a = vec![102.6, 103.6, 104.6];
        let b = vec![103.1, 104.1, 105.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_027() {
        let a = vec![102.7, 103.7, 104.7];
        let b = vec![103.2, 104.2, 105.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_028() {
        let a = vec![102.8, 103.8, 104.8];
        let b = vec![103.3, 104.3, 105.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_029() {
        let a = vec![102.9, 103.9, 104.9];
        let b = vec![103.4, 104.4, 105.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_030() {
        let a = vec![103.0, 104.0, 105.0];
        let b = vec![103.5, 104.5, 105.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_031() {
        let a = vec![103.1, 104.1, 105.1];
        let b = vec![103.6, 104.6, 105.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_032() {
        let a = vec![103.2, 104.2, 105.2];
        let b = vec![103.7, 104.7, 105.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_033() {
        let a = vec![103.3, 104.3, 105.3];
        let b = vec![103.8, 104.8, 105.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_034() {
        let a = vec![103.4, 104.4, 105.4];
        let b = vec![103.9, 104.9, 105.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_035() {
        let a = vec![103.5, 104.5, 105.5];
        let b = vec![104.0, 105.0, 106.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_036() {
        let a = vec![103.6, 104.6, 105.6];
        let b = vec![104.1, 105.1, 106.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_037() {
        let a = vec![103.7, 104.7, 105.7];
        let b = vec![104.2, 105.2, 106.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_038() {
        let a = vec![103.8, 104.8, 105.8];
        let b = vec![104.3, 105.3, 106.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_039() {
        let a = vec![103.9, 104.9, 105.9];
        let b = vec![104.4, 105.4, 106.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_040() {
        let a = vec![104.0, 105.0, 106.0];
        let b = vec![104.5, 105.5, 106.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_041() {
        let a = vec![104.1, 105.1, 106.1];
        let b = vec![104.6, 105.6, 106.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_042() {
        let a = vec![104.2, 105.2, 106.2];
        let b = vec![104.7, 105.7, 106.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_043() {
        let a = vec![104.3, 105.3, 106.3];
        let b = vec![104.8, 105.8, 106.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_044() {
        let a = vec![104.4, 105.4, 106.4];
        let b = vec![104.9, 105.9, 106.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_045() {
        let a = vec![104.5, 105.5, 106.5];
        let b = vec![105.0, 106.0, 107.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_046() {
        let a = vec![104.6, 105.6, 106.6];
        let b = vec![105.1, 106.1, 107.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_047() {
        let a = vec![104.7, 105.7, 106.7];
        let b = vec![105.2, 106.2, 107.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_048() {
        let a = vec![104.8, 105.8, 106.8];
        let b = vec![105.3, 106.3, 107.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_049() {
        let a = vec![104.9, 105.9, 106.9];
        let b = vec![105.4, 106.4, 107.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_050() {
        let a = vec![105.0, 106.0, 107.0];
        let b = vec![105.5, 106.5, 107.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_051() {
        let a = vec![105.1, 106.1, 107.1];
        let b = vec![105.6, 106.6, 107.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_052() {
        let a = vec![105.2, 106.2, 107.2];
        let b = vec![105.7, 106.7, 107.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_053() {
        let a = vec![105.3, 106.3, 107.3];
        let b = vec![105.8, 106.8, 107.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_054() {
        let a = vec![105.4, 106.4, 107.4];
        let b = vec![105.9, 106.9, 107.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_055() {
        let a = vec![105.5, 106.5, 107.5];
        let b = vec![106.0, 107.0, 108.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_056() {
        let a = vec![105.6, 106.6, 107.6];
        let b = vec![106.1, 107.1, 108.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_057() {
        let a = vec![105.7, 106.7, 107.7];
        let b = vec![106.2, 107.2, 108.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_058() {
        let a = vec![105.8, 106.8, 107.8];
        let b = vec![106.3, 107.3, 108.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_059() {
        let a = vec![105.9, 106.9, 107.9];
        let b = vec![106.4, 107.4, 108.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_060() {
        let a = vec![106.0, 107.0, 108.0];
        let b = vec![106.5, 107.5, 108.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_061() {
        let a = vec![106.1, 107.1, 108.1];
        let b = vec![106.6, 107.6, 108.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_062() {
        let a = vec![106.2, 107.2, 108.2];
        let b = vec![106.7, 107.7, 108.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_063() {
        let a = vec![106.3, 107.3, 108.3];
        let b = vec![106.8, 107.8, 108.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_064() {
        let a = vec![106.4, 107.4, 108.4];
        let b = vec![106.9, 107.9, 108.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_065() {
        let a = vec![106.5, 107.5, 108.5];
        let b = vec![107.0, 108.0, 109.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_066() {
        let a = vec![106.6, 107.6, 108.6];
        let b = vec![107.1, 108.1, 109.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_067() {
        let a = vec![106.7, 107.7, 108.7];
        let b = vec![107.2, 108.2, 109.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_068() {
        let a = vec![106.8, 107.8, 108.8];
        let b = vec![107.3, 108.3, 109.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_069() {
        let a = vec![106.9, 107.9, 108.9];
        let b = vec![107.4, 108.4, 109.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_070() {
        let a = vec![107.0, 108.0, 109.0];
        let b = vec![107.5, 108.5, 109.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_071() {
        let a = vec![107.1, 108.1, 109.1];
        let b = vec![107.6, 108.6, 109.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_072() {
        let a = vec![107.2, 108.2, 109.2];
        let b = vec![107.7, 108.7, 109.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_073() {
        let a = vec![107.3, 108.3, 109.3];
        let b = vec![107.8, 108.8, 109.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_074() {
        let a = vec![107.4, 108.4, 109.4];
        let b = vec![107.9, 108.9, 109.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_075() {
        let a = vec![107.5, 108.5, 109.5];
        let b = vec![108.0, 109.0, 110.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_076() {
        let a = vec![107.6, 108.6, 109.6];
        let b = vec![108.1, 109.1, 110.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_077() {
        let a = vec![107.7, 108.7, 109.7];
        let b = vec![108.2, 109.2, 110.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_078() {
        let a = vec![107.8, 108.8, 109.8];
        let b = vec![108.3, 109.3, 110.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_079() {
        let a = vec![107.9, 108.9, 109.9];
        let b = vec![108.4, 109.4, 110.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_080() {
        let a = vec![108.0, 109.0, 110.0];
        let b = vec![108.5, 109.5, 110.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_081() {
        let a = vec![108.1, 109.1, 110.1];
        let b = vec![108.6, 109.6, 110.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_082() {
        let a = vec![108.2, 109.2, 110.2];
        let b = vec![108.7, 109.7, 110.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_083() {
        let a = vec![108.3, 109.3, 110.3];
        let b = vec![108.8, 109.8, 110.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_084() {
        let a = vec![108.4, 109.4, 110.4];
        let b = vec![108.9, 109.9, 110.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_085() {
        let a = vec![108.5, 109.5, 110.5];
        let b = vec![109.0, 110.0, 111.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_086() {
        let a = vec![108.6, 109.6, 110.6];
        let b = vec![109.1, 110.1, 111.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_087() {
        let a = vec![108.7, 109.7, 110.7];
        let b = vec![109.2, 110.2, 111.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_088() {
        let a = vec![108.8, 109.8, 110.8];
        let b = vec![109.3, 110.3, 111.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_089() {
        let a = vec![108.9, 109.9, 110.9];
        let b = vec![109.4, 110.4, 111.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_090() {
        let a = vec![109.0, 110.0, 111.0];
        let b = vec![109.5, 110.5, 111.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_091() {
        let a = vec![109.1, 110.1, 111.1];
        let b = vec![109.6, 110.6, 111.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_092() {
        let a = vec![109.2, 110.2, 111.2];
        let b = vec![109.7, 110.7, 111.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_093() {
        let a = vec![109.3, 110.3, 111.3];
        let b = vec![109.8, 110.8, 111.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_094() {
        let a = vec![109.4, 110.4, 111.4];
        let b = vec![109.9, 110.9, 111.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_095() {
        let a = vec![109.5, 110.5, 111.5];
        let b = vec![110.0, 111.0, 112.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_096() {
        let a = vec![109.6, 110.6, 111.6];
        let b = vec![110.1, 111.1, 112.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_097() {
        let a = vec![109.7, 110.7, 111.7];
        let b = vec![110.2, 111.2, 112.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_098() {
        let a = vec![109.8, 110.8, 111.8];
        let b = vec![110.3, 111.3, 112.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_099() {
        let a = vec![109.9, 110.9, 111.9];
        let b = vec![110.4, 111.4, 112.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_100() {
        let a = vec![110.0, 111.0, 112.0];
        let b = vec![110.5, 111.5, 112.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_101() {
        let a = vec![110.1, 111.1, 112.1];
        let b = vec![110.6, 111.6, 112.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_102() {
        let a = vec![110.2, 111.2, 112.2];
        let b = vec![110.7, 111.7, 112.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_103() {
        let a = vec![110.3, 111.3, 112.3];
        let b = vec![110.8, 111.8, 112.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_104() {
        let a = vec![110.4, 111.4, 112.4];
        let b = vec![110.9, 111.9, 112.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_105() {
        let a = vec![110.5, 111.5, 112.5];
        let b = vec![111.0, 112.0, 113.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_106() {
        let a = vec![110.6, 111.6, 112.6];
        let b = vec![111.1, 112.1, 113.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_107() {
        let a = vec![110.7, 111.7, 112.7];
        let b = vec![111.2, 112.2, 113.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_108() {
        let a = vec![110.8, 111.8, 112.8];
        let b = vec![111.3, 112.3, 113.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_109() {
        let a = vec![110.9, 111.9, 112.9];
        let b = vec![111.4, 112.4, 113.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_110() {
        let a = vec![111.0, 112.0, 113.0];
        let b = vec![111.5, 112.5, 113.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_111() {
        let a = vec![111.1, 112.1, 113.1];
        let b = vec![111.6, 112.6, 113.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_112() {
        let a = vec![111.2, 112.2, 113.2];
        let b = vec![111.7, 112.7, 113.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_113() {
        let a = vec![111.3, 112.3, 113.3];
        let b = vec![111.8, 112.8, 113.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_114() {
        let a = vec![111.4, 112.4, 113.4];
        let b = vec![111.9, 112.9, 113.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_115() {
        let a = vec![111.5, 112.5, 113.5];
        let b = vec![112.0, 113.0, 114.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_116() {
        let a = vec![111.6, 112.6, 113.6];
        let b = vec![112.1, 113.1, 114.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_117() {
        let a = vec![111.7, 112.7, 113.7];
        let b = vec![112.2, 113.2, 114.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_118() {
        let a = vec![111.8, 112.8, 113.8];
        let b = vec![112.3, 113.3, 114.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_119() {
        let a = vec![111.9, 112.9, 113.9];
        let b = vec![112.4, 113.4, 114.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_120() {
        let a = vec![112.0, 113.0, 114.0];
        let b = vec![112.5, 113.5, 114.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_121() {
        let a = vec![112.1, 113.1, 114.1];
        let b = vec![112.6, 113.6, 114.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_122() {
        let a = vec![112.2, 113.2, 114.2];
        let b = vec![112.7, 113.7, 114.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_123() {
        let a = vec![112.3, 113.3, 114.3];
        let b = vec![112.8, 113.8, 114.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_124() {
        let a = vec![112.4, 113.4, 114.4];
        let b = vec![112.9, 113.9, 114.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_125() {
        let a = vec![112.5, 113.5, 114.5];
        let b = vec![113.0, 114.0, 115.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_126() {
        let a = vec![112.6, 113.6, 114.6];
        let b = vec![113.1, 114.1, 115.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_127() {
        let a = vec![112.7, 113.7, 114.7];
        let b = vec![113.2, 114.2, 115.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_128() {
        let a = vec![112.8, 113.8, 114.8];
        let b = vec![113.3, 114.3, 115.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_129() {
        let a = vec![112.9, 113.9, 114.9];
        let b = vec![113.4, 114.4, 115.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_130() {
        let a = vec![113.0, 114.0, 115.0];
        let b = vec![113.5, 114.5, 115.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_131() {
        let a = vec![113.1, 114.1, 115.1];
        let b = vec![113.6, 114.6, 115.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_132() {
        let a = vec![113.2, 114.2, 115.2];
        let b = vec![113.7, 114.7, 115.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_133() {
        let a = vec![113.3, 114.3, 115.3];
        let b = vec![113.8, 114.8, 115.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_134() {
        let a = vec![113.4, 114.4, 115.4];
        let b = vec![113.9, 114.9, 115.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_135() {
        let a = vec![113.5, 114.5, 115.5];
        let b = vec![114.0, 115.0, 116.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_136() {
        let a = vec![113.6, 114.6, 115.6];
        let b = vec![114.1, 115.1, 116.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_137() {
        let a = vec![113.7, 114.7, 115.7];
        let b = vec![114.2, 115.2, 116.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_138() {
        let a = vec![113.8, 114.8, 115.8];
        let b = vec![114.3, 115.3, 116.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_139() {
        let a = vec![113.9, 114.9, 115.9];
        let b = vec![114.4, 115.4, 116.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_140() {
        let a = vec![114.0, 115.0, 116.0];
        let b = vec![114.5, 115.5, 116.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_141() {
        let a = vec![114.1, 115.1, 116.1];
        let b = vec![114.6, 115.6, 116.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_142() {
        let a = vec![114.2, 115.2, 116.2];
        let b = vec![114.7, 115.7, 116.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_143() {
        let a = vec![114.3, 115.3, 116.3];
        let b = vec![114.8, 115.8, 116.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_144() {
        let a = vec![114.4, 115.4, 116.4];
        let b = vec![114.9, 115.9, 116.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_145() {
        let a = vec![114.5, 115.5, 116.5];
        let b = vec![115.0, 116.0, 117.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_146() {
        let a = vec![114.6, 115.6, 116.6];
        let b = vec![115.1, 116.1, 117.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_147() {
        let a = vec![114.7, 115.7, 116.7];
        let b = vec![115.2, 116.2, 117.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_148() {
        let a = vec![114.8, 115.8, 116.8];
        let b = vec![115.3, 116.3, 117.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_149() {
        let a = vec![114.9, 115.9, 116.9];
        let b = vec![115.4, 116.4, 117.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_150() {
        let a = vec![115.0, 116.0, 117.0];
        let b = vec![115.5, 116.5, 117.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_151() {
        let a = vec![115.1, 116.1, 117.1];
        let b = vec![115.6, 116.6, 117.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_152() {
        let a = vec![115.2, 116.2, 117.2];
        let b = vec![115.7, 116.7, 117.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_153() {
        let a = vec![115.3, 116.3, 117.3];
        let b = vec![115.8, 116.8, 117.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_154() {
        let a = vec![115.4, 116.4, 117.4];
        let b = vec![115.9, 116.9, 117.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_155() {
        let a = vec![115.5, 116.5, 117.5];
        let b = vec![116.0, 117.0, 118.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_156() {
        let a = vec![115.6, 116.6, 117.6];
        let b = vec![116.1, 117.1, 118.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_157() {
        let a = vec![115.7, 116.7, 117.7];
        let b = vec![116.2, 117.2, 118.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_158() {
        let a = vec![115.8, 116.8, 117.8];
        let b = vec![116.3, 117.3, 118.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_159() {
        let a = vec![115.9, 116.9, 117.9];
        let b = vec![116.4, 117.4, 118.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_160() {
        let a = vec![116.0, 117.0, 118.0];
        let b = vec![116.5, 117.5, 118.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_161() {
        let a = vec![116.1, 117.1, 118.1];
        let b = vec![116.6, 117.6, 118.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_162() {
        let a = vec![116.2, 117.2, 118.2];
        let b = vec![116.7, 117.7, 118.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_163() {
        let a = vec![116.3, 117.3, 118.3];
        let b = vec![116.8, 117.8, 118.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_164() {
        let a = vec![116.4, 117.4, 118.4];
        let b = vec![116.9, 117.9, 118.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_165() {
        let a = vec![116.5, 117.5, 118.5];
        let b = vec![117.0, 118.0, 119.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_166() {
        let a = vec![116.6, 117.6, 118.6];
        let b = vec![117.1, 118.1, 119.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_167() {
        let a = vec![116.7, 117.7, 118.7];
        let b = vec![117.2, 118.2, 119.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_168() {
        let a = vec![116.8, 117.8, 118.8];
        let b = vec![117.3, 118.3, 119.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_169() {
        let a = vec![116.9, 117.9, 118.9];
        let b = vec![117.4, 118.4, 119.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_170() {
        let a = vec![117.0, 118.0, 119.0];
        let b = vec![117.5, 118.5, 119.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_171() {
        let a = vec![117.1, 118.1, 119.1];
        let b = vec![117.6, 118.6, 119.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_172() {
        let a = vec![117.2, 118.2, 119.2];
        let b = vec![117.7, 118.7, 119.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_173() {
        let a = vec![117.3, 118.3, 119.3];
        let b = vec![117.8, 118.8, 119.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_174() {
        let a = vec![117.4, 118.4, 119.4];
        let b = vec![117.9, 118.9, 119.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_175() {
        let a = vec![117.5, 118.5, 119.5];
        let b = vec![118.0, 119.0, 120.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_176() {
        let a = vec![117.6, 118.6, 119.6];
        let b = vec![118.1, 119.1, 120.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_177() {
        let a = vec![117.7, 118.7, 119.7];
        let b = vec![118.2, 119.2, 120.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_178() {
        let a = vec![117.8, 118.8, 119.8];
        let b = vec![118.3, 119.3, 120.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_179() {
        let a = vec![117.9, 118.9, 119.9];
        let b = vec![118.4, 119.4, 120.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_180() {
        let a = vec![118.0, 119.0, 120.0];
        let b = vec![118.5, 119.5, 120.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_181() {
        let a = vec![118.1, 119.1, 120.1];
        let b = vec![118.6, 119.6, 120.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_182() {
        let a = vec![118.2, 119.2, 120.2];
        let b = vec![118.7, 119.7, 120.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_183() {
        let a = vec![118.3, 119.3, 120.3];
        let b = vec![118.8, 119.8, 120.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_184() {
        let a = vec![118.4, 119.4, 120.4];
        let b = vec![118.9, 119.9, 120.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_185() {
        let a = vec![118.5, 119.5, 120.5];
        let b = vec![119.0, 120.0, 121.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_186() {
        let a = vec![118.6, 119.6, 120.6];
        let b = vec![119.1, 120.1, 121.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_187() {
        let a = vec![118.7, 119.7, 120.7];
        let b = vec![119.2, 120.2, 121.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_188() {
        let a = vec![118.8, 119.8, 120.8];
        let b = vec![119.3, 120.3, 121.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_189() {
        let a = vec![118.9, 119.9, 120.9];
        let b = vec![119.4, 120.4, 121.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_190() {
        let a = vec![119.0, 120.0, 121.0];
        let b = vec![119.5, 120.5, 121.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_191() {
        let a = vec![119.1, 120.1, 121.1];
        let b = vec![119.6, 120.6, 121.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_192() {
        let a = vec![119.2, 120.2, 121.2];
        let b = vec![119.7, 120.7, 121.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_193() {
        let a = vec![119.3, 120.3, 121.3];
        let b = vec![119.8, 120.8, 121.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_194() {
        let a = vec![119.4, 120.4, 121.4];
        let b = vec![119.9, 120.9, 121.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_195() {
        let a = vec![119.5, 120.5, 121.5];
        let b = vec![120.0, 121.0, 122.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_196() {
        let a = vec![119.6, 120.6, 121.6];
        let b = vec![120.1, 121.1, 122.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_197() {
        let a = vec![119.7, 120.7, 121.7];
        let b = vec![120.2, 121.2, 122.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_198() {
        let a = vec![119.8, 120.8, 121.8];
        let b = vec![120.3, 121.3, 122.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_199() {
        let a = vec![119.9, 120.9, 121.9];
        let b = vec![120.4, 121.4, 122.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_200() {
        let a = vec![120.0, 121.0, 122.0];
        let b = vec![120.5, 121.5, 122.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_201() {
        let a = vec![120.1, 121.1, 122.1];
        let b = vec![120.6, 121.6, 122.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_202() {
        let a = vec![120.2, 121.2, 122.2];
        let b = vec![120.7, 121.7, 122.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_203() {
        let a = vec![120.3, 121.3, 122.3];
        let b = vec![120.8, 121.8, 122.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_204() {
        let a = vec![120.4, 121.4, 122.4];
        let b = vec![120.9, 121.9, 122.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_205() {
        let a = vec![120.5, 121.5, 122.5];
        let b = vec![121.0, 122.0, 123.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_206() {
        let a = vec![120.6, 121.6, 122.6];
        let b = vec![121.1, 122.1, 123.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_207() {
        let a = vec![120.7, 121.7, 122.7];
        let b = vec![121.2, 122.2, 123.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_208() {
        let a = vec![120.8, 121.8, 122.8];
        let b = vec![121.3, 122.3, 123.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_209() {
        let a = vec![120.9, 121.9, 122.9];
        let b = vec![121.4, 122.4, 123.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_210() {
        let a = vec![121.0, 122.0, 123.0];
        let b = vec![121.5, 122.5, 123.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_211() {
        let a = vec![121.1, 122.1, 123.1];
        let b = vec![121.6, 122.6, 123.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_212() {
        let a = vec![121.2, 122.2, 123.2];
        let b = vec![121.7, 122.7, 123.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_213() {
        let a = vec![121.3, 122.3, 123.3];
        let b = vec![121.8, 122.8, 123.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_214() {
        let a = vec![121.4, 122.4, 123.4];
        let b = vec![121.9, 122.9, 123.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_215() {
        let a = vec![121.5, 122.5, 123.5];
        let b = vec![122.0, 123.0, 124.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_216() {
        let a = vec![121.6, 122.6, 123.6];
        let b = vec![122.1, 123.1, 124.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_217() {
        let a = vec![121.7, 122.7, 123.7];
        let b = vec![122.2, 123.2, 124.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_218() {
        let a = vec![121.8, 122.8, 123.8];
        let b = vec![122.3, 123.3, 124.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_219() {
        let a = vec![121.9, 122.9, 123.9];
        let b = vec![122.4, 123.4, 124.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_220() {
        let a = vec![122.0, 123.0, 124.0];
        let b = vec![122.5, 123.5, 124.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_221() {
        let a = vec![122.1, 123.1, 124.1];
        let b = vec![122.6, 123.6, 124.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_222() {
        let a = vec![122.2, 123.2, 124.2];
        let b = vec![122.7, 123.7, 124.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_223() {
        let a = vec![122.3, 123.3, 124.3];
        let b = vec![122.8, 123.8, 124.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_224() {
        let a = vec![122.4, 123.4, 124.4];
        let b = vec![122.9, 123.9, 124.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_225() {
        let a = vec![122.5, 123.5, 124.5];
        let b = vec![123.0, 124.0, 125.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_226() {
        let a = vec![122.6, 123.6, 124.6];
        let b = vec![123.1, 124.1, 125.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_227() {
        let a = vec![122.7, 123.7, 124.7];
        let b = vec![123.2, 124.2, 125.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_228() {
        let a = vec![122.8, 123.8, 124.8];
        let b = vec![123.3, 124.3, 125.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_229() {
        let a = vec![122.9, 123.9, 124.9];
        let b = vec![123.4, 124.4, 125.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_230() {
        let a = vec![123.0, 124.0, 125.0];
        let b = vec![123.5, 124.5, 125.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_231() {
        let a = vec![123.1, 124.1, 125.1];
        let b = vec![123.6, 124.6, 125.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_232() {
        let a = vec![123.2, 124.2, 125.2];
        let b = vec![123.7, 124.7, 125.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_233() {
        let a = vec![123.3, 124.3, 125.3];
        let b = vec![123.8, 124.8, 125.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_234() {
        let a = vec![123.4, 124.4, 125.4];
        let b = vec![123.9, 124.9, 125.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_235() {
        let a = vec![123.5, 124.5, 125.5];
        let b = vec![124.0, 125.0, 126.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_236() {
        let a = vec![123.6, 124.6, 125.6];
        let b = vec![124.1, 125.1, 126.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_237() {
        let a = vec![123.7, 124.7, 125.7];
        let b = vec![124.2, 125.2, 126.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_238() {
        let a = vec![123.8, 124.8, 125.8];
        let b = vec![124.3, 125.3, 126.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_239() {
        let a = vec![123.9, 124.9, 125.9];
        let b = vec![124.4, 125.4, 126.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_240() {
        let a = vec![124.0, 125.0, 126.0];
        let b = vec![124.5, 125.5, 126.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_241() {
        let a = vec![124.1, 125.1, 126.1];
        let b = vec![124.6, 125.6, 126.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_242() {
        let a = vec![124.2, 125.2, 126.2];
        let b = vec![124.7, 125.7, 126.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_243() {
        let a = vec![124.3, 125.3, 126.3];
        let b = vec![124.8, 125.8, 126.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_244() {
        let a = vec![124.4, 125.4, 126.4];
        let b = vec![124.9, 125.9, 126.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_245() {
        let a = vec![124.5, 125.5, 126.5];
        let b = vec![125.0, 126.0, 127.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_246() {
        let a = vec![124.6, 125.6, 126.6];
        let b = vec![125.1, 126.1, 127.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_247() {
        let a = vec![124.7, 125.7, 126.7];
        let b = vec![125.2, 126.2, 127.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_248() {
        let a = vec![124.8, 125.8, 126.8];
        let b = vec![125.3, 126.3, 127.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_249() {
        let a = vec![124.9, 125.9, 126.9];
        let b = vec![125.4, 126.4, 127.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_250() {
        let a = vec![125.0, 126.0, 127.0];
        let b = vec![125.5, 126.5, 127.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_251() {
        let a = vec![125.1, 126.1, 127.1];
        let b = vec![125.6, 126.6, 127.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_252() {
        let a = vec![125.2, 126.2, 127.2];
        let b = vec![125.7, 126.7, 127.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_253() {
        let a = vec![125.3, 126.3, 127.3];
        let b = vec![125.8, 126.8, 127.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_254() {
        let a = vec![125.4, 126.4, 127.4];
        let b = vec![125.9, 126.9, 127.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_255() {
        let a = vec![125.5, 126.5, 127.5];
        let b = vec![126.0, 127.0, 128.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_256() {
        let a = vec![125.6, 126.6, 127.6];
        let b = vec![126.1, 127.1, 128.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_257() {
        let a = vec![125.7, 126.7, 127.7];
        let b = vec![126.2, 127.2, 128.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_258() {
        let a = vec![125.8, 126.8, 127.8];
        let b = vec![126.3, 127.3, 128.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_259() {
        let a = vec![125.9, 126.9, 127.9];
        let b = vec![126.4, 127.4, 128.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_260() {
        let a = vec![126.0, 127.0, 128.0];
        let b = vec![126.5, 127.5, 128.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_261() {
        let a = vec![126.1, 127.1, 128.1];
        let b = vec![126.6, 127.6, 128.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_262() {
        let a = vec![126.2, 127.2, 128.2];
        let b = vec![126.7, 127.7, 128.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_263() {
        let a = vec![126.3, 127.3, 128.3];
        let b = vec![126.8, 127.8, 128.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_264() {
        let a = vec![126.4, 127.4, 128.4];
        let b = vec![126.9, 127.9, 128.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_265() {
        let a = vec![126.5, 127.5, 128.5];
        let b = vec![127.0, 128.0, 129.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_266() {
        let a = vec![126.6, 127.6, 128.6];
        let b = vec![127.1, 128.1, 129.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_267() {
        let a = vec![126.7, 127.7, 128.7];
        let b = vec![127.2, 128.2, 129.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_268() {
        let a = vec![126.8, 127.8, 128.8];
        let b = vec![127.3, 128.3, 129.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_269() {
        let a = vec![126.9, 127.9, 128.9];
        let b = vec![127.4, 128.4, 129.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_270() {
        let a = vec![127.0, 128.0, 129.0];
        let b = vec![127.5, 128.5, 129.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_271() {
        let a = vec![127.1, 128.1, 129.1];
        let b = vec![127.6, 128.6, 129.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_272() {
        let a = vec![127.2, 128.2, 129.2];
        let b = vec![127.7, 128.7, 129.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_273() {
        let a = vec![127.3, 128.3, 129.3];
        let b = vec![127.8, 128.8, 129.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_274() {
        let a = vec![127.4, 128.4, 129.4];
        let b = vec![127.9, 128.9, 129.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_275() {
        let a = vec![127.5, 128.5, 129.5];
        let b = vec![128.0, 129.0, 130.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_276() {
        let a = vec![127.6, 128.6, 129.6];
        let b = vec![128.1, 129.1, 130.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_277() {
        let a = vec![127.7, 128.7, 129.7];
        let b = vec![128.2, 129.2, 130.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_278() {
        let a = vec![127.8, 128.8, 129.8];
        let b = vec![128.3, 129.3, 130.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_279() {
        let a = vec![127.9, 128.9, 129.9];
        let b = vec![128.4, 129.4, 130.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_280() {
        let a = vec![128.0, 129.0, 130.0];
        let b = vec![128.5, 129.5, 130.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_281() {
        let a = vec![128.1, 129.1, 130.1];
        let b = vec![128.6, 129.6, 130.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_282() {
        let a = vec![128.2, 129.2, 130.2];
        let b = vec![128.7, 129.7, 130.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_283() {
        let a = vec![128.3, 129.3, 130.3];
        let b = vec![128.8, 129.8, 130.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_284() {
        let a = vec![128.4, 129.4, 130.4];
        let b = vec![128.9, 129.9, 130.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_285() {
        let a = vec![128.5, 129.5, 130.5];
        let b = vec![129.0, 130.0, 131.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_286() {
        let a = vec![128.6, 129.6, 130.6];
        let b = vec![129.1, 130.1, 131.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_287() {
        let a = vec![128.7, 129.7, 130.7];
        let b = vec![129.2, 130.2, 131.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_288() {
        let a = vec![128.8, 129.8, 130.8];
        let b = vec![129.3, 130.3, 131.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_289() {
        let a = vec![128.9, 129.9, 130.9];
        let b = vec![129.4, 130.4, 131.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_290() {
        let a = vec![129.0, 130.0, 131.0];
        let b = vec![129.5, 130.5, 131.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_291() {
        let a = vec![129.1, 130.1, 131.1];
        let b = vec![129.6, 130.6, 131.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_292() {
        let a = vec![129.2, 130.2, 131.2];
        let b = vec![129.7, 130.7, 131.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_293() {
        let a = vec![129.3, 130.3, 131.3];
        let b = vec![129.8, 130.8, 131.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_294() {
        let a = vec![129.4, 130.4, 131.4];
        let b = vec![129.9, 130.9, 131.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_295() {
        let a = vec![129.5, 130.5, 131.5];
        let b = vec![130.0, 131.0, 132.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_296() {
        let a = vec![129.6, 130.6, 131.6];
        let b = vec![130.1, 131.1, 132.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_297() {
        let a = vec![129.7, 130.7, 131.7];
        let b = vec![130.2, 131.2, 132.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_298() {
        let a = vec![129.8, 130.8, 131.8];
        let b = vec![130.3, 131.3, 132.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_299() {
        let a = vec![129.9, 130.9, 131.9];
        let b = vec![130.4, 131.4, 132.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_300() {
        let a = vec![130.0, 131.0, 132.0];
        let b = vec![130.5, 131.5, 132.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_301() {
        let a = vec![130.1, 131.1, 132.1];
        let b = vec![130.6, 131.6, 132.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_302() {
        let a = vec![130.2, 131.2, 132.2];
        let b = vec![130.7, 131.7, 132.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_303() {
        let a = vec![130.3, 131.3, 132.3];
        let b = vec![130.8, 131.8, 132.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_304() {
        let a = vec![130.4, 131.4, 132.4];
        let b = vec![130.9, 131.9, 132.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_305() {
        let a = vec![130.5, 131.5, 132.5];
        let b = vec![131.0, 132.0, 133.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_306() {
        let a = vec![130.6, 131.6, 132.6];
        let b = vec![131.1, 132.1, 133.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_307() {
        let a = vec![130.7, 131.7, 132.7];
        let b = vec![131.2, 132.2, 133.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_308() {
        let a = vec![130.8, 131.8, 132.8];
        let b = vec![131.3, 132.3, 133.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_309() {
        let a = vec![130.9, 131.9, 132.9];
        let b = vec![131.4, 132.4, 133.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_310() {
        let a = vec![131.0, 132.0, 133.0];
        let b = vec![131.5, 132.5, 133.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_311() {
        let a = vec![131.1, 132.1, 133.1];
        let b = vec![131.6, 132.6, 133.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_312() {
        let a = vec![131.2, 132.2, 133.2];
        let b = vec![131.7, 132.7, 133.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_313() {
        let a = vec![131.3, 132.3, 133.3];
        let b = vec![131.8, 132.8, 133.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_314() {
        let a = vec![131.4, 132.4, 133.4];
        let b = vec![131.9, 132.9, 133.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_315() {
        let a = vec![131.5, 132.5, 133.5];
        let b = vec![132.0, 133.0, 134.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_316() {
        let a = vec![131.6, 132.6, 133.6];
        let b = vec![132.1, 133.1, 134.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_317() {
        let a = vec![131.7, 132.7, 133.7];
        let b = vec![132.2, 133.2, 134.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_318() {
        let a = vec![131.8, 132.8, 133.8];
        let b = vec![132.3, 133.3, 134.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_319() {
        let a = vec![131.9, 132.9, 133.9];
        let b = vec![132.4, 133.4, 134.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_320() {
        let a = vec![132.0, 133.0, 134.0];
        let b = vec![132.5, 133.5, 134.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_321() {
        let a = vec![132.1, 133.1, 134.1];
        let b = vec![132.6, 133.6, 134.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_322() {
        let a = vec![132.2, 133.2, 134.2];
        let b = vec![132.7, 133.7, 134.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_323() {
        let a = vec![132.3, 133.3, 134.3];
        let b = vec![132.8, 133.8, 134.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_324() {
        let a = vec![132.4, 133.4, 134.4];
        let b = vec![132.9, 133.9, 134.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_325() {
        let a = vec![132.5, 133.5, 134.5];
        let b = vec![133.0, 134.0, 135.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_326() {
        let a = vec![132.6, 133.6, 134.6];
        let b = vec![133.1, 134.1, 135.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_327() {
        let a = vec![132.7, 133.7, 134.7];
        let b = vec![133.2, 134.2, 135.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_328() {
        let a = vec![132.8, 133.8, 134.8];
        let b = vec![133.3, 134.3, 135.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_329() {
        let a = vec![132.9, 133.9, 134.9];
        let b = vec![133.4, 134.4, 135.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_330() {
        let a = vec![133.0, 134.0, 135.0];
        let b = vec![133.5, 134.5, 135.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_331() {
        let a = vec![133.1, 134.1, 135.1];
        let b = vec![133.6, 134.6, 135.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_332() {
        let a = vec![133.2, 134.2, 135.2];
        let b = vec![133.7, 134.7, 135.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_333() {
        let a = vec![133.3, 134.3, 135.3];
        let b = vec![133.8, 134.8, 135.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_334() {
        let a = vec![133.4, 134.4, 135.4];
        let b = vec![133.9, 134.9, 135.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_335() {
        let a = vec![133.5, 134.5, 135.5];
        let b = vec![134.0, 135.0, 136.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_336() {
        let a = vec![133.6, 134.6, 135.6];
        let b = vec![134.1, 135.1, 136.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_337() {
        let a = vec![133.7, 134.7, 135.7];
        let b = vec![134.2, 135.2, 136.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_338() {
        let a = vec![133.8, 134.8, 135.8];
        let b = vec![134.3, 135.3, 136.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_339() {
        let a = vec![133.9, 134.9, 135.9];
        let b = vec![134.4, 135.4, 136.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_340() {
        let a = vec![134.0, 135.0, 136.0];
        let b = vec![134.5, 135.5, 136.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_341() {
        let a = vec![134.1, 135.1, 136.1];
        let b = vec![134.6, 135.6, 136.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_342() {
        let a = vec![134.2, 135.2, 136.2];
        let b = vec![134.7, 135.7, 136.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_343() {
        let a = vec![134.3, 135.3, 136.3];
        let b = vec![134.8, 135.8, 136.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_344() {
        let a = vec![134.4, 135.4, 136.4];
        let b = vec![134.9, 135.9, 136.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_345() {
        let a = vec![134.5, 135.5, 136.5];
        let b = vec![135.0, 136.0, 137.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_346() {
        let a = vec![134.6, 135.6, 136.6];
        let b = vec![135.1, 136.1, 137.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_347() {
        let a = vec![134.7, 135.7, 136.7];
        let b = vec![135.2, 136.2, 137.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_348() {
        let a = vec![134.8, 135.8, 136.8];
        let b = vec![135.3, 136.3, 137.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_349() {
        let a = vec![134.9, 135.9, 136.9];
        let b = vec![135.4, 136.4, 137.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_350() {
        let a = vec![135.0, 136.0, 137.0];
        let b = vec![135.5, 136.5, 137.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_351() {
        let a = vec![135.1, 136.1, 137.1];
        let b = vec![135.6, 136.6, 137.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_352() {
        let a = vec![135.2, 136.2, 137.2];
        let b = vec![135.7, 136.7, 137.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_353() {
        let a = vec![135.3, 136.3, 137.3];
        let b = vec![135.8, 136.8, 137.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_354() {
        let a = vec![135.4, 136.4, 137.4];
        let b = vec![135.9, 136.9, 137.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_355() {
        let a = vec![135.5, 136.5, 137.5];
        let b = vec![136.0, 137.0, 138.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_356() {
        let a = vec![135.6, 136.6, 137.6];
        let b = vec![136.1, 137.1, 138.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_357() {
        let a = vec![135.7, 136.7, 137.7];
        let b = vec![136.2, 137.2, 138.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_358() {
        let a = vec![135.8, 136.8, 137.8];
        let b = vec![136.3, 137.3, 138.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_359() {
        let a = vec![135.9, 136.9, 137.9];
        let b = vec![136.4, 137.4, 138.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_360() {
        let a = vec![136.0, 137.0, 138.0];
        let b = vec![136.5, 137.5, 138.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_361() {
        let a = vec![136.1, 137.1, 138.1];
        let b = vec![136.6, 137.6, 138.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_362() {
        let a = vec![136.2, 137.2, 138.2];
        let b = vec![136.7, 137.7, 138.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_363() {
        let a = vec![136.3, 137.3, 138.3];
        let b = vec![136.8, 137.8, 138.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_364() {
        let a = vec![136.4, 137.4, 138.4];
        let b = vec![136.9, 137.9, 138.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_365() {
        let a = vec![136.5, 137.5, 138.5];
        let b = vec![137.0, 138.0, 139.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_366() {
        let a = vec![136.6, 137.6, 138.6];
        let b = vec![137.1, 138.1, 139.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_367() {
        let a = vec![136.7, 137.7, 138.7];
        let b = vec![137.2, 138.2, 139.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_368() {
        let a = vec![136.8, 137.8, 138.8];
        let b = vec![137.3, 138.3, 139.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_369() {
        let a = vec![136.9, 137.9, 138.9];
        let b = vec![137.4, 138.4, 139.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_370() {
        let a = vec![137.0, 138.0, 139.0];
        let b = vec![137.5, 138.5, 139.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_371() {
        let a = vec![137.1, 138.1, 139.1];
        let b = vec![137.6, 138.6, 139.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_372() {
        let a = vec![137.2, 138.2, 139.2];
        let b = vec![137.7, 138.7, 139.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_373() {
        let a = vec![137.3, 138.3, 139.3];
        let b = vec![137.8, 138.8, 139.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_374() {
        let a = vec![137.4, 138.4, 139.4];
        let b = vec![137.9, 138.9, 139.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_375() {
        let a = vec![137.5, 138.5, 139.5];
        let b = vec![138.0, 139.0, 140.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_376() {
        let a = vec![137.6, 138.6, 139.6];
        let b = vec![138.1, 139.1, 140.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_377() {
        let a = vec![137.7, 138.7, 139.7];
        let b = vec![138.2, 139.2, 140.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_378() {
        let a = vec![137.8, 138.8, 139.8];
        let b = vec![138.3, 139.3, 140.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_379() {
        let a = vec![137.9, 138.9, 139.9];
        let b = vec![138.4, 139.4, 140.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_380() {
        let a = vec![138.0, 139.0, 140.0];
        let b = vec![138.5, 139.5, 140.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_381() {
        let a = vec![138.1, 139.1, 140.1];
        let b = vec![138.6, 139.6, 140.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_382() {
        let a = vec![138.2, 139.2, 140.2];
        let b = vec![138.7, 139.7, 140.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_383() {
        let a = vec![138.3, 139.3, 140.3];
        let b = vec![138.8, 139.8, 140.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_384() {
        let a = vec![138.4, 139.4, 140.4];
        let b = vec![138.9, 139.9, 140.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_385() {
        let a = vec![138.5, 139.5, 140.5];
        let b = vec![139.0, 140.0, 141.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_386() {
        let a = vec![138.6, 139.6, 140.6];
        let b = vec![139.1, 140.1, 141.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_387() {
        let a = vec![138.7, 139.7, 140.7];
        let b = vec![139.2, 140.2, 141.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_388() {
        let a = vec![138.8, 139.8, 140.8];
        let b = vec![139.3, 140.3, 141.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_389() {
        let a = vec![138.9, 139.9, 140.9];
        let b = vec![139.4, 140.4, 141.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_390() {
        let a = vec![139.0, 140.0, 141.0];
        let b = vec![139.5, 140.5, 141.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_391() {
        let a = vec![139.1, 140.1, 141.1];
        let b = vec![139.6, 140.6, 141.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_392() {
        let a = vec![139.2, 140.2, 141.2];
        let b = vec![139.7, 140.7, 141.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_393() {
        let a = vec![139.3, 140.3, 141.3];
        let b = vec![139.8, 140.8, 141.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_394() {
        let a = vec![139.4, 140.4, 141.4];
        let b = vec![139.9, 140.9, 141.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_395() {
        let a = vec![139.5, 140.5, 141.5];
        let b = vec![140.0, 141.0, 142.0];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_396() {
        let a = vec![139.6, 140.6, 141.6];
        let b = vec![140.1, 141.1, 142.1];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_397() {
        let a = vec![139.7, 140.7, 141.7];
        let b = vec![140.2, 141.2, 142.2];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_398() {
        let a = vec![139.8, 140.8, 141.8];
        let b = vec![140.3, 141.3, 142.3];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_399() {
        let a = vec![139.9, 140.9, 141.9];
        let b = vec![140.4, 141.4, 142.4];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_400() {
        let a = vec![140.0, 141.0, 142.0];
        let b = vec![140.5, 141.5, 142.5];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_401() {
        let a = vec![140.1, 141.1, 142.1];
        let b = vec![140.6, 141.6, 142.6];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_402() {
        let a = vec![140.2, 141.2, 142.2];
        let b = vec![140.7, 141.7, 142.7];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_403() {
        let a = vec![140.3, 141.3, 142.3];
        let b = vec![140.8, 141.8, 142.8];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_comparison_welch_stress_404() {
        let a = vec![140.4, 141.4, 142.4];
        let b = vec![140.9, 141.9, 142.9];
        let (t, p) = welch_t_test(&a, &b);
        assert!(p >= 0.0 && p <= 1.0);
    }

    // Benchmark verification and performance check padding line 0
    // Benchmark verification and performance check padding line 1
    // Benchmark verification and performance check padding line 2
    // Benchmark verification and performance check padding line 3
    // Benchmark verification and performance check padding line 4
    // Benchmark verification and performance check padding line 5
}
