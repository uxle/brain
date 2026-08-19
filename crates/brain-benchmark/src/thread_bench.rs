//! # Multi-Threaded Scalability & Contention Analysis
//!
//! Evaluates speedup ratios, parallel efficiency, and lock contention across worker thread sweeps.

use crate::core::BenchResult;

/// Scalability report summarizing parallel speedups.
#[derive(Debug, Clone)]
pub struct ScalabilityReport {
    pub name: String,
    pub thread_counts: Vec<usize>,
    pub mean_durations_ns: Vec<f64>,
    pub speedup_ratios: Vec<f64>,
    pub parallel_efficiency: Vec<f64>,
}

impl ScalabilityReport {
    /// Evaluates scalability from a list of per-thread benchmark results.
    pub fn from_results(name: impl Into<String>, results: &[(usize, BenchResult)]) -> Self {
        let mut thread_counts = Vec::new();
        let mut mean_durations_ns = Vec::new();
        let mut speedup_ratios = Vec::new();
        let mut parallel_efficiency = Vec::new();

        let base_mean = results.first().map(|(_, r)| r.mean_nanos()).unwrap_or(1.0);

        for &(threads, ref r) in results {
            let mean = r.mean_nanos();
            let speedup = if mean > 0.0 { base_mean / mean } else { 1.0 };
            let efficiency = speedup / threads.max(1) as f64;

            thread_counts.push(threads);
            mean_durations_ns.push(mean);
            speedup_ratios.push(speedup);
            parallel_efficiency.push(efficiency);
        }

        Self {
            name: name.into(),
            thread_counts,
            mean_durations_ns,
            speedup_ratios,
            parallel_efficiency,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
