//! # Core Benchmarking Configuration and Result Types
//!
//! Defines the [`Benchmark`] trait, [`BenchConfig`], [`BenchResult`], [`Sample`],
//! and measurement units.

use std::time::Duration;

/// Measurement units for benchmark metrics and throughput reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeasurementUnit {
    #[default]
    Nanoseconds,
    Microseconds,
    Milliseconds,
    Seconds,
    OperationsPerSecond,
    GigaFlops,
    GigabytesPerSecond,
    Custom(&'static str),
}

impl MeasurementUnit {
    /// Returns the standard display symbol for the unit.
    pub fn symbol(&self) -> &'static str {
        match self {
            MeasurementUnit::Nanoseconds => "ns",
            MeasurementUnit::Microseconds => "µs",
            MeasurementUnit::Milliseconds => "ms",
            MeasurementUnit::Seconds => "s",
            MeasurementUnit::OperationsPerSecond => "ops/s",
            MeasurementUnit::GigaFlops => "GFLOPS",
            MeasurementUnit::GigabytesPerSecond => "GB/s",
            MeasurementUnit::Custom(s) => s,
        }
    }
}

/// Strategy for iterating the benchmark target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IterationStrategy {
    /// Execute a fixed number of iterations per sample.
    FixedIterations(usize),
    /// Execute iterations until a fixed duration target is met.
    FixedDuration(Duration),
    /// Adaptive iteration discovery aiming for a target sample duration.
    Adaptive {
        target_sample_duration: Duration,
        max_iterations: usize,
    },
}

impl Default for IterationStrategy {
    fn default() -> Self {
        Self::Adaptive {
            target_sample_duration: Duration::from_millis(50),
            max_iterations: 1_000_000,
        }
    }
}

/// Complete configuration options for a benchmark execution.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Descriptive name of the benchmark.
    pub name: String,
    /// Number of warmup iterations executed prior to measurement.
    pub warmup_iterations: usize,
    /// Warmup duration target.
    pub warmup_time: Duration,
    /// Measurement duration target.
    pub measurement_time: Duration,
    /// Number of samples to collect.
    pub sample_count: usize,
    /// Number of worker threads for parallel benchmarks.
    pub num_threads: usize,
    /// Category tags for filtering.
    pub tags: Vec<String>,
    /// Iteration strategy.
    pub strategy: IterationStrategy,
    /// Unit of measurement.
    pub unit: MeasurementUnit,
    /// Operations count per iteration (for FLOPS/Throughput calculation).
    pub ops_per_iteration: u64,
    /// Bytes processed per iteration.
    pub bytes_per_iteration: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            name: "unnamed_bench".to_string(),
            warmup_iterations: 10,
            warmup_time: Duration::from_millis(100),
            measurement_time: Duration::from_secs(1),
            sample_count: 50,
            num_threads: 1,
            tags: Vec::new(),
            strategy: IterationStrategy::default(),
            unit: MeasurementUnit::Nanoseconds,
            ops_per_iteration: 1,
            bytes_per_iteration: 0,
        }
    }
}

impl BenchConfig {
    /// Creates a new `BenchConfig` with the specified name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Sets the sample count.
    pub fn with_sample_count(mut self, samples: usize) -> Self {
        self.sample_count = samples.max(1);
        self
    }

    /// Sets warmup iterations.
    pub fn with_warmup_iterations(mut self, iters: usize) -> Self {
        self.warmup_iterations = iters;
        self
    }

    /// Sets measurement time.
    pub fn with_measurement_time(mut self, duration: Duration) -> Self {
        self.measurement_time = duration;
        self
    }

    /// Appends a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets FLOPs per iteration.
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.ops_per_iteration = flops;
        self.unit = MeasurementUnit::GigaFlops;
        self
    }

    /// Sets bytes processed per iteration.
    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes_per_iteration = bytes;
        self.unit = MeasurementUnit::GigabytesPerSecond;
        self
    }
}

/// Represents a single measurement sample containing elapsed time and iteration count.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Measured execution duration for the batch.
    pub duration: Duration,
    /// Number of iterations executed in this batch.
    pub iterations: usize,
    /// Average duration per single iteration.
    pub time_per_iteration: Duration,
}

impl Sample {
    /// Creates a new `Sample`.
    pub fn new(duration: Duration, iterations: usize) -> Self {
        let iters = iterations.max(1);
        let time_per_iteration = duration / iters as u32;
        Self {
            duration,
            iterations: iters,
            time_per_iteration,
        }
    }
}

/// The complete outcome of a benchmark execution including raw samples and metadata.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Benchmark configuration used for the run.
    pub config: BenchConfig,
    /// Collected measurement samples.
    pub samples: Vec<Sample>,
    /// Total wall-clock time spent on the benchmark run.
    pub total_wall_time: Duration,
    /// Raw nanoseconds per iteration for statistical processing.
    pub raw_nanos: Vec<f64>,
}

impl BenchResult {
    /// Creates a new `BenchResult` from raw samples.
    pub fn new(config: BenchConfig, samples: Vec<Sample>, total_wall_time: Duration) -> Self {
        let raw_nanos: Vec<f64> = samples
            .iter()
            .map(|s| s.time_per_iteration.as_nanos() as f64)
            .collect();

        Self {
            config,
            samples,
            total_wall_time,
            raw_nanos,
        }
    }

    /// Computes summary statistics over all collected samples.
    pub fn statistics(&self) -> crate::statistics::Statistics {
        crate::statistics::Statistics::compute(&self.raw_nanos)
    }

    /// Returns the mean iteration duration in nanoseconds.
    pub fn mean_nanos(&self) -> f64 {
        self.statistics().mean
    }

    /// Returns the median iteration duration in nanoseconds.
    pub fn median_nanos(&self) -> f64 {
        self.statistics().median
    }

    /// Returns estimated throughput in GFLOPS if ops_per_iteration was configured.
    pub fn gflops(&self) -> f64 {
        let mean_secs = self.mean_nanos() / 1_000_000_000.0;
        if mean_secs <= 0.0 {
            0.0
        } else {
            (self.config.ops_per_iteration as f64 / 1_000_000_000.0) / mean_secs
        }
    }

    /// Returns estimated throughput in GB/s if bytes_per_iteration was configured.
    pub fn gigabytes_per_second(&self) -> f64 {
        let mean_secs = self.mean_nanos() / 1_000_000_000.0;
        if mean_secs <= 0.0 {
            0.0
        } else {
            (self.config.bytes_per_iteration as f64 / 1_000_000_000.0) / mean_secs
        }
    }
}

/// The core benchmark execution trait.
pub trait Benchmark: Send + Sync {
    /// Returns the descriptive name of the benchmark.
    fn name(&self) -> &str;
    /// Returns configuration for this benchmark.
    fn config(&self) -> &BenchConfig;
    /// Executes the benchmark and returns the collected results.
    fn run(&mut self) -> brain_core::BrainResult<BenchResult>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
