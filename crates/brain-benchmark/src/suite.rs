//! # Benchmark Suite Organization and Orchestration
//!
//! Provides [`BenchmarkSuite`] for registering benchmarks, applying tag/name filters,
//! ordering executions, and batch evaluation.

use crate::core::{BenchResult, Benchmark};
use brain_core::BrainResult;

/// A registry and execution coordinator for a collection of benchmarks.
#[derive(Default)]
pub struct BenchmarkSuite {
    name: String,
    benchmarks: Vec<Box<dyn Benchmark>>,
}

impl BenchmarkSuite {
    /// Creates a new `BenchmarkSuite`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            benchmarks: Vec::new(),
        }
    }

    /// Returns the descriptive name of the suite.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Registers a benchmark into the suite.
    pub fn add(&mut self, bench: Box<dyn Benchmark>) -> &mut Self {
        self.benchmarks.push(bench);
        self
    }

    /// Returns the number of registered benchmarks.
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    /// Returns whether the suite is empty.
    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }

    /// Executes all benchmarks in the suite sequentially.
    pub fn run_all(&mut self) -> BrainResult<Vec<BenchResult>> {
        let mut results = Vec::with_capacity(self.benchmarks.len());
        for bench in &mut self.benchmarks {
            results.push(bench.run()?);
        }
        Ok(results)
    }

    /// Executes only benchmarks matching the given name substring.
    pub fn run_filtered<F>(&mut self, mut filter: F) -> BrainResult<Vec<BenchResult>>
    where
        F: FnMut(&str, &[String]) -> bool,
    {
        let mut results = Vec::new();
        for bench in &mut self.benchmarks {
            let name = bench.name();
            let tags = &bench.config().tags;
            if filter(name, tags) {
                results.push(bench.run()?);
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
