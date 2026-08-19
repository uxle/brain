//! # Brain Benchmark — High-Precision Deep Learning Performance Suite
//!
//! Production-grade performance engineering, high-resolution timing,
//! statistical analysis, A/B regression detection, hardware probing, and multi-format reporting.
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_benchmark::prelude::*;
//!
//! let mut timer = Timer::new();
//! timer.start();
//! let _ = std::hint::black_box(2 + 2);
//! let elapsed = timer.stop();
//! assert!(elapsed.as_nanos() >= 0);
//! ```
//!
//! ## Running a Benchmark
//!
//! ```rust
//! use brain_benchmark::prelude::*;
//!
//! let config = BenchConfig::new("example_add").with_sample_count(5);
//! let result = Runner::run_benchmark(&config, || {
//!     let _ = std::hint::black_box(42 * 42);
//! }).unwrap();
//! assert_eq!(result.samples.len(), 5);
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity
)]

pub mod baseline;
pub mod bench_macros;
pub mod compare;
pub mod core;
pub mod distribution;
pub mod energy;
pub mod export;
pub mod graph_bench;
pub mod hardware;
pub mod histogram;
pub mod r#impl;
pub mod io_bench;
pub mod kernels;
pub mod memory_bench;
pub mod models;
pub mod ops;
pub mod profiler;
pub mod report;
pub mod runner;
pub mod statistics;
pub mod suite;
pub mod thread_bench;
pub mod timer;
pub mod utils;

// Re-exports
pub use core::{BenchConfig, BenchResult, Benchmark, IterationStrategy, MeasurementUnit, Sample};
pub use r#impl::{FnBenchmark, KernelBenchmark, ModelBenchmark};
pub use runner::Runner;
pub use statistics::Statistics;
pub use suite::BenchmarkSuite;
pub use timer::{calibrate_timer_overhead, BenchTimerGuard, Timer};

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_benchmark::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_benchmark::version_string;
/// assert_eq!(version_string(), "brain-benchmark v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-benchmark v{}", VERSION)
}

/// Runs the standard benchmark suite across all registered operations.
pub fn run_all() -> brain_core::BrainResult<Vec<BenchResult>> {
    let mut suite = BenchmarkSuite::default();
    suite.run_all()
}

/// Common prelude imports for convenient benchmarking.
///
/// ```rust
/// use brain_benchmark::prelude::*;
/// let timer = Timer::new();
/// assert!(!timer.is_running());
/// ```
pub mod prelude {
    pub use crate::core::{BenchConfig, BenchResult, Benchmark, IterationStrategy, MeasurementUnit, Sample};
    pub use crate::r#impl::{FnBenchmark, KernelBenchmark, ModelBenchmark};
    pub use crate::runner::Runner;
    pub use crate::statistics::Statistics;
    pub use crate::suite::BenchmarkSuite;
    pub use crate::timer::{BenchTimerGuard, Timer};
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
