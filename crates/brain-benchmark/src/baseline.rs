//! # Baseline Storage & CI Regression Gating
//!
//! Manages saving and loading benchmark baseline records in JSON format and
//! enforcing CI performance regression budgets.

use crate::compare::{compare_runs, ComparisonResult, ComparisonVerdict};
use crate::core::BenchResult;
use brain_core::{BrainError, BrainResult};
use std::collections::HashMap;

/// Stores baseline benchmark runs for comparison.
#[derive(Debug, Clone, Default)]
pub struct BaselineStore {
    runs: HashMap<String, BenchResult>,
}

impl BaselineStore {
    /// Creates a new empty `BaselineStore`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a benchmark run to the store.
    pub fn add(&mut self, result: BenchResult) {
        self.runs.insert(result.config.name.clone(), result);
    }

    /// Gets a baseline run by name.
    pub fn get(&self, name: &str) -> Option<&BenchResult> {
        self.runs.get(name)
    }

    /// Evaluates current benchmark runs against stored baselines.
    pub fn evaluate_regressions(
        &self,
        current_runs: &[BenchResult],
        max_regression_pct: f64,
    ) -> BrainResult<Vec<ComparisonResult>> {
        let mut comparisons = Vec::new();
        let mut failures = Vec::new();

        for current in current_runs {
            if let Some(base) = self.get(&current.config.name) {
                let comp = compare_runs(base, current, 0.05);
                if comp.verdict == ComparisonVerdict::Regression && comp.percent_change > max_regression_pct {
                    failures.push(format!(
                        "Benchmark '{}' regressed by {:.2}% (threshold: {:.2}%)",
                        comp.name, comp.percent_change, max_regression_pct
                    ));
                }
                comparisons.push(comp);
            }
        }

        if !failures.is_empty() {
            Err(BrainError::invalid_value(failures.join("; ")))
        } else {
            Ok(comparisons)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
