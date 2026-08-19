//! # Brain Distributed — Enterprise-Grade Distributed Machine Learning Suite
//!
//! Features NCCL-like collective operations (Ring/Tree AllReduce, AllGather, Broadcast),
//! Data Parallelism (`DataParallel`), Pipeline Parallelism (1F1B schedule), Tensor Parallelism,
//! gradient compression (Top-K, Quantized, EF21), fault tolerance, and communication backends.
//!
//! ## Subsystems
//!
//! * [`collective`] - Collective primitives (AllReduce, Reduce, AllGather, Broadcast, Scatter)
//! * [`comm`] - Transport backends (`MemComm`, `TcpComm`) and message framing
//! * [`group`] - Process groups (`ProcessGroup`) and communicators
//! * [`data_parallel`] - `DataParallel` gradient synchronization
//! * [`model_parallel`] - Model parallelism and inter-stage activation routing
//! * [`pipeline`] - 1F1B pipelined schedule execution and micro-batching
//! * [`grad_allreduce`] - Gradient bucketing and overlap engines
//! * [`grad_compression`] - Top-K sparsification and error feedback (EF21)
//! * [`tensor_parallel`] - Tensor parallelism (Row/Column sharded linear layers)
//! * [`sync`] - Barrier synchronization and clocks
//! * [`fault`] - Node failure detection and retry policies
//! * [`topology`] - Topology mapping and shared-memory optimizations
//! * [`cluster`] - Cluster node management and discovery
//! * [`async_exec`] - Async collective task graph scheduler
//! * [`bench`] - Bandwidth and latency communication benchmarks
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_distributed::prelude::*;
//!
//! let ctx = DistributedContext::new(0, 4);
//! assert!(ctx.is_master());
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

pub mod async_exec;
pub mod bench;
pub mod builder;
pub mod cluster;
pub mod collective;
pub mod comm;
pub mod compute;
pub mod config;
pub mod core;
pub mod data_parallel;
pub mod fault;
pub mod grad_allreduce;
pub mod grad_compression;
pub mod group;
pub mod helper;
pub mod r#impl;
pub mod model_parallel;
pub mod ops;
pub mod pipeline;
pub mod process;
pub mod sync;
pub mod tensor_parallel;
pub mod topology;
pub mod transform;
pub mod utils;

// Re-exports
pub use builder::DistributedBuilder;
pub use config::{BackendKind, DistributedConfig};
pub use core::{DistributedContext, Rank, WorldSize};
pub use data_parallel::DataParallel;
pub use group::ProcessGroup;

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_distributed::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_distributed::version_string;
/// assert_eq!(version_string(), "brain-distributed v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-distributed v{}", VERSION)
}

/// Standard prelude imports for distributed training.
///
/// ```rust
/// use brain_distributed::prelude::*;
/// let ctx = DistributedContext::new(1, 4);
/// assert_eq!(ctx.rank, 1);
/// ```
pub mod prelude {
    pub use crate::builder::DistributedBuilder;
    pub use crate::config::{BackendKind, DistributedConfig};
    pub use crate::core::{DistributedContext, Rank, WorldSize};
    pub use crate::data_parallel::DataParallel;
    pub use crate::group::ProcessGroup;
    pub use brain_core::Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
