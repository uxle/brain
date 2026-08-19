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
}
