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
}
