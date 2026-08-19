//! # Brain Data — High-Throughput Deep Learning Data Pipelines
//!
//! Features multi-stage asynchronous pipelines, batching, collation, distributed sharding,
//! memory-safe shuffling, caching, prefetching, and tensor adapters.
//!
//! ## Subsystems
//!
//! * [`core`] - `Sample`, `SampleBatch`, `DataSource`, and `DataReader`
//! * [`pipeline`] - Composable pipeline builder and stage runner
//! * [`stages`] - `MapStage`, `FilterStage`, `BatchStage`, `ShuffleStage`, `PrefetchStage`
//! * [`collate`] - `default_collate`, `pad_collate`, and custom collation functions
//! * [`batch`] - `BatchIter`, `EpochIter`, and batch samplers
//! * [`loading`] - `FileLoader`, `MemoryLoader`, `TensorLoader`
//! * [`streaming`] - Chunked streaming datasets for massive corpora
//! * [`caching`] - LRU memory caching and checksum-validated disk cache
//! * [`shuffle`] - Index permutation shuffle and window shuffling
//! * [`samplers`] - Sequential, Random, Weighted, and Distributed samplers
//! * [`prefetch`] - Asynchronous multi-threaded prefetch buffers
//! * [`errors`] - Pipeline error handling and retry policies
//! * [`metrics`] - Throughput meters and latency tracking
//! * [`checkpoint`] - Pipeline epoch/sample state recovery
//! * [`mmap`] - Memory-mapped binary chunk reading
//! * [`compression`] - Pure-Rust RLE, Delta, and Variable-Byte compression
//! * [`multi`] - Multi-source zip, concat, and interleave pipelines
//! * [`profile`] - Per-stage profiling and bottleneck detectors
//! * [`backpressure`] - Channel flow control and watermarks
//! * [`lazy`] - Lazy feature evaluation and memoization
//! * [`interop`] - Interop with `brain-dataset` and `brain-core::Tensor`
//! * [`config`] - Global pipeline configuration and thread pools
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_data::prelude::*;
//!
//! let sample = Sample::new(0, Tensor::zeros(vec![2, 2]));
//! assert_eq!(sample.id, 0);
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

pub mod backpressure;
pub mod batch;
pub mod caching;
pub mod checkpoint;
pub mod collate;
pub mod compression;
pub mod config;
pub mod core;
pub mod errors;
pub mod r#impl;
pub mod interop;
pub mod lazy;
pub mod loading;
pub mod metrics;
pub mod mmap;
pub mod multi;
pub mod ops;
pub mod pipeline;
pub mod prefetch;
pub mod profile;
pub mod samplers;
pub mod shuffle;
pub mod stages;
pub mod streaming;
pub mod utils;

// Re-exports
pub use core::{DataReader, DataSource, Sample, SampleBatch};
pub use pipeline::Pipeline;
pub use r#impl::PipelineRunner;

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_data::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_data::version_string;
/// assert_eq!(version_string(), "brain-data v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-data v{}", VERSION)
}

/// Standard prelude imports for data pipelines.
///
/// ```rust
/// use brain_data::prelude::*;
/// let sample = Sample::new(1, Tensor::zeros(vec![1]));
/// assert_eq!(sample.id, 1);
/// ```
pub mod prelude {
    pub use crate::core::{DataReader, DataSource, Sample, SampleBatch};
    pub use crate::pipeline::Pipeline;
    pub use crate::r#impl::PipelineRunner;
    pub use brain_core::Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
