//! # Concrete Benchmark Implementations
//!
//! Provides [`FnBenchmark`], [`ModelBenchmark`], [`KernelBenchmark`], and parametric wrappers.

use crate::core::{BenchConfig, BenchResult, Benchmark};
use crate::runner::Runner;
use brain_core::BrainResult;

/// Benchmark implementation wrapping a Rust closure.
pub struct FnBenchmark<F> {
    config: BenchConfig,
    func: F,
}

impl<F: FnMut() + Send + Sync + Clone> FnBenchmark<F> {
    /// Creates a new `FnBenchmark` from configuration and closure.
    pub fn new(config: BenchConfig, func: F) -> Self {
        Self { config, func }
    }

    /// Creates a new `FnBenchmark` with default configuration.
    pub fn from_fn(name: impl Into<String>, func: F) -> Self {
        Self::new(BenchConfig::new(name), func)
    }
}

impl<F: FnMut() + Send + Sync + Clone> Benchmark for FnBenchmark<F> {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn config(&self) -> &BenchConfig {
        &self.config
    }

    fn run(&mut self) -> BrainResult<BenchResult> {
        let f = self.func.clone();
        Runner::run_benchmark(&self.config, f)
    }
}

/// Benchmark wrapper specifically for neural network forward and backward passes.
pub struct ModelBenchmark<F> {
    config: BenchConfig,
    forward_fn: F,
    batch_size: usize,
}

impl<F: FnMut() + Send + Sync + Clone> ModelBenchmark<F> {
    /// Creates a new `ModelBenchmark`.
    pub fn new(name: impl Into<String>, batch_size: usize, forward_fn: F) -> Self {
        let config = BenchConfig::new(name).with_tag("model");
        Self {
            config,
            forward_fn,
            batch_size,
        }
    }

    /// Returns the batch size configured for this model benchmark.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl<F: FnMut() + Send + Sync + Clone> Benchmark for ModelBenchmark<F> {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn config(&self) -> &BenchConfig {
        &self.config
    }

    fn run(&mut self) -> BrainResult<BenchResult> {
        let f = self.forward_fn.clone();
        Runner::run_benchmark(&self.config, f)
    }
}

/// Benchmark wrapper for low-level compute kernels with FLOP counting.
pub struct KernelBenchmark<F> {
    config: BenchConfig,
    kernel_fn: F,
    flop_count: u64,
}

impl<F: FnMut() + Send + Sync + Clone> KernelBenchmark<F> {
    /// Creates a new `KernelBenchmark` with configured FLOP count.
    pub fn new(name: impl Into<String>, flop_count: u64, kernel_fn: F) -> Self {
        let config = BenchConfig::new(name)
            .with_flops(flop_count)
            .with_tag("kernel");
        Self {
            config,
            kernel_fn,
            flop_count,
        }
    }

    /// Returns the configured FLOP count per run.
    pub fn flop_count(&self) -> u64 {
        self.flop_count
    }
}

impl<F: FnMut() + Send + Sync + Clone> Benchmark for KernelBenchmark<F> {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn config(&self) -> &BenchConfig {
        &self.config
    }

    fn run(&mut self) -> BrainResult<BenchResult> {
        let f = self.kernel_fn.clone();
        Runner::run_benchmark(&self.config, f)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
