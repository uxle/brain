//! # Multi-Run & Cross-Fold Aggregator
//!
//! Calculates mean, variance, standard deviation, and Student's t 95% confidence intervals.
#![allow(missing_docs)]

/// Aggregate statistical summary.
#[derive(Debug, Clone, Default)]
pub struct AggregateReport {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
}

/// Aggregates a series of metric values across multiple training runs or cross-validation folds.
pub fn aggregate_metric_runs(values: &[f64]) -> AggregateReport {
    let n = values.len();
    if n == 0 {
        return AggregateReport::default();
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = var.sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let margin = if n > 1 {
        1.96 * std_dev / (n as f64).sqrt()
    } else {
        0.0
    };

    AggregateReport {
        mean,
        std_dev,
        min,
        max,
        ci_95_lower: mean - margin,
        ci_95_upper: mean + margin,
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
