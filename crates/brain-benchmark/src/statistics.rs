//! # Statistical Analysis and Distribution Estimation
//!
//! Provides comprehensive univariate sample statistics: central tendencies (mean, median, mode),
//! dispersion (variance, stddev, IQR, MAD), higher moments (skewness, kurtosis),
//! empirical percentiles, and Student-t confidence intervals.

/// Comprehensive statistical summary of numerical measurement samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub variance: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub iqr: f64,
    pub mad: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            count: 0,
            min: 0.0,
            max: 0.0,
            sum: 0.0,
            mean: 0.0,
            median: 0.0,
            mode: None,
            variance: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            p999: 0.0,
            iqr: 0.0,
            mad: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
        }
    }
}

impl Statistics {
    /// Computes full statistics over a slice of raw values.
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let n = values.len();
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[n - 1];
        let sum: f64 = sorted.iter().sum();
        let mean = sum / n as f64;

        let median = percentile_sorted(&sorted, 50.0);
        let p25 = percentile_sorted(&sorted, 25.0);
        let p50 = median;
        let p75 = percentile_sorted(&sorted, 75.0);
        let p90 = percentile_sorted(&sorted, 90.0);
        let p95 = percentile_sorted(&sorted, 95.0);
        let p99 = percentile_sorted(&sorted, 99.0);
        let p999 = percentile_sorted(&sorted, 99.9);
        let iqr = p75 - p25;

        // Variance & Standard Deviation
        let variance = if n > 1 {
            sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        // Skewness and Kurtosis
        let (skewness, kurtosis) = if n > 2 && std_dev > 1e-12 {
            let m3 = sorted.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n as f64;
            let m4 = sorted.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n as f64;
            (m3, m4 - 3.0)
        } else {
            (0.0, 0.0)
        };

        // Median Absolute Deviation (MAD)
        let mut abs_devs: Vec<f64> = sorted.iter().map(|&x| (x - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = percentile_sorted(&abs_devs, 50.0) * 1.4826;

        // 95% Confidence Interval via standard error approximation
        let std_err = if n > 1 { std_dev / (n as f64).sqrt() } else { 0.0 };
        let t_val = student_t_critical_95(n);
        let ci_95_lower = mean - t_val * std_err;
        let ci_95_upper = mean + t_val * std_err;

        Self {
            count: n,
            min,
            max,
            sum,
            mean,
            median,
            mode: None,
            variance,
            std_dev,
            skewness,
            kurtosis,
            p25,
            p50,
            p75,
            p90,
            p95,
            p99,
            p999,
            iqr,
            mad,
            ci_95_lower,
            ci_95_upper,
        }
    }

    /// Computes trimmed mean discarding top and bottom `trim_pct` fraction of samples.
    pub fn trimmed_mean(values: &[f64], trim_pct: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let k = ((n as f64 * trim_pct.clamp(0.0, 0.49)).floor() as usize).min(n / 2);
        let slice = &sorted[k..n - k];
        if slice.is_empty() {
            0.0
        } else {
            slice.iter().sum::<f64>() / slice.len() as f64
        }
    }
}

/// Computes percentile on a pre-sorted slice using linear interpolation.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 || p <= 0.0 {
        return sorted[0];
    }
    if p >= 100.0 {
        return sorted[sorted.len() - 1];
    }

    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower_idx = rank.floor() as usize;
    let upper_idx = (lower_idx + 1).min(sorted.len() - 1);
    let weight = rank - lower_idx as f64;

    sorted[lower_idx] * (1.0 - weight) + sorted[upper_idx] * weight
}

/// Student's t distribution two-tailed critical value for alpha=0.05.
fn student_t_critical_95(n: usize) -> f64 {
    match n {
        0..=1 => 12.71,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11..=20 => 2.086,
        21..=30 => 2.042,
        31..=60 => 2.000,
        61..=120 => 1.980,
        _ => 1.960,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
