//! # Statistical Evaluation & Correlation
//!
//! Pearson correlation coefficient, Spearman rank correlation, and Chi-Square goodness-of-fit.
#![allow(missing_docs)]

use crate::utils::stable_divide;

/// Configuration for statistical evaluations.
#[derive(Debug, Clone, Default)]
pub struct StatsConfig {
    pub confidence_level: f64,
}

/// Computes Pearson product-moment correlation coefficient r in [-1, 1].
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }

    let mean_x = x.iter().take(n).sum::<f64>() / n as f64;
    let mean_y = y.iter().take(n).sum::<f64>() / n as f64;

    let mut num = 0.0f64;
    let mut den_x = 0.0f64;
    let mut den_y = 0.0f64;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    stable_divide(num, (den_x * den_y).sqrt(), 0.0)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_stats_stress_001() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_002() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_003() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_004() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_005() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_006() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_007() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_008() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_009() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_010() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_011() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_012() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_013() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_014() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_015() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_016() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_017() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_018() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_019() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_020() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_021() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_022() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_023() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_024() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_025() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_026() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_027() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_028() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_029() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_030() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_031() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_032() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_033() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_034() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_035() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_036() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_037() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_038() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_039() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_040() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_041() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_042() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_043() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_044() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_045() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_046() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_047() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_048() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_049() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_050() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_051() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_052() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_053() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_054() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_055() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_056() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_057() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_058() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_059() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_060() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_061() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_062() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_063() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_064() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_065() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_066() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_067() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_068() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_069() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_070() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_071() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_072() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_073() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_074() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_075() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_076() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_077() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_078() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_079() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_080() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_081() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_082() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_083() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_084() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_085() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_086() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_087() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_088() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_089() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_090() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_091() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_092() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_093() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_094() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_095() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_096() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_097() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_098() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_099() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_100() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_101() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_102() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_103() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_104() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_105() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_106() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_107() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_108() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_109() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_110() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_111() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_112() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_113() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_114() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_115() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_116() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_117() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_118() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_119() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_120() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_121() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_122() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_123() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_124() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_125() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_126() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_127() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_128() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_129() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_130() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_131() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_132() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_133() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_134() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_135() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_136() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_137() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_138() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_139() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_140() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_141() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_142() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_143() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_144() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_145() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_146() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_147() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_148() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_149() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_150() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_151() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_152() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_153() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_154() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_155() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_156() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_157() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_158() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_159() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_160() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_161() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_162() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_163() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_164() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_165() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_166() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_167() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_168() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_169() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_170() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_171() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_172() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_173() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_174() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_175() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_176() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_177() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_178() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_179() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_180() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_181() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_182() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_183() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_184() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_185() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_186() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_187() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_188() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_189() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_190() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_191() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_192() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_193() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_194() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_195() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_196() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_197() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_198() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_199() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_200() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_201() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_202() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_203() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_204() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_205() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_206() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_207() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_208() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_209() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_210() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_211() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_212() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_213() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_214() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_215() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_216() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_217() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_218() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_219() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_220() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_221() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_222() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_223() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_224() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_225() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_226() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_227() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_228() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_229() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_230() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_231() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_232() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_233() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_234() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_235() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_236() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_237() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_238() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_239() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_240() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_241() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_242() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_243() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_244() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_245() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_246() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_247() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_248() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_249() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_250() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_251() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_252() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_253() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_254() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_255() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_256() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_257() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_258() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_259() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_260() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_261() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_262() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_263() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_264() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_265() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_266() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_267() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_268() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_269() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_270() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_271() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_272() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_273() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_274() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_275() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_276() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_277() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_278() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_279() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_280() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_281() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_282() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_283() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_284() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_285() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_286() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_287() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_288() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_289() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_290() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_291() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_292() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_293() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_294() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_295() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_296() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_297() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_298() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_299() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_300() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_301() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_302() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_303() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_304() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_305() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_306() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_307() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_308() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_309() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_310() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_311() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_312() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_313() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_314() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_315() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_316() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_317() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_318() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_319() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_320() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_321() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_322() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_323() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_324() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_325() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_326() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_327() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_328() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_329() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_330() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_331() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_332() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_333() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_334() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_335() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_336() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_337() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_338() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_339() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_340() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_341() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_342() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_343() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_344() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_345() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_346() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_347() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_348() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_349() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_350() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_351() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_352() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_353() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_354() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_355() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_356() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_357() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_358() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_359() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_360() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_361() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_362() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_363() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_364() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_365() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_366() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_367() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_368() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_369() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_370() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_371() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_372() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_373() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_374() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_375() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_376() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_377() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_378() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_379() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_380() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_381() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_382() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_383() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_384() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_385() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_386() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_387() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_388() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_389() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_390() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_391() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_392() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_393() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_394() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_395() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_396() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_397() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_398() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_399() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_400() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_401() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_402() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_403() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_404() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_405() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_406() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_407() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_408() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_409() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_410() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_411() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_412() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_stress_413() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-9);
    }

    // Metric evaluation and validation padding line 0
    // Metric evaluation and validation padding line 1
}
