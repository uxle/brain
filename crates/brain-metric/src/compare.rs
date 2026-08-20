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
    let rel = if score_b.abs() > 1e-12 {
        (delta / score_b) * 100.0
    } else {
        0.0
    };
    CompareReport {
        delta,
        relative_gain_pct: rel,
        is_model_a_better: delta > 0.0,
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
