//! # Brain Dataset — Comprehensive Machine Learning Dataset Ecosystem
//!
//! Provides dataset abstractions, vision/text/audio generators, transforms, samplers, and multi-worker loaders.
//!
//! ## Subsystems
//!
//! * [`dataset`] - `Dataset` trait, `MapDataset`, `ConcatDataset`, vision, text, audio, and tabular generators
//! * [`transforms`] - Vision, text, audio, and numeric transform pipelines
//! * [`samplers`] - Sequential, random, weighted, and subset samplers
//! * [`loaders`] - Multi-worker `DataLoader` and `WorkerPool`
//! * [`splits`] - Train/val/test and k-fold cross-validation dataset splitting
//! * [`cache`] - In-memory and disk-backed dataset caching
//! * [`statistics`] - Dataset feature metrics and class distribution analysis
//! * [`inspect`] - Dataset structure and corruption inspector
//! * [`balance`] - Class rebalancing and SMOTE synthetic generation
//! * [`stream`] - Streaming CSV and text dataset parsers
//! * [`builder`] - Fluent `DatasetBuilder` interface
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_dataset::prelude::*;
//!
//! let item = Item::new(0, Tensor::zeros(vec![28, 28]));
//! assert_eq!(item.id, 0);
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

pub mod analyze;
pub mod balance;
pub mod builder;
pub mod cache;
pub mod compute;
pub mod config;
pub mod core;
pub mod dataset;
pub mod helper;
pub mod r#impl;
pub mod inspect;
pub mod loaders;
pub mod manage;
pub mod ops;
pub mod optimize;
pub mod process;
pub mod samplers;
pub mod splits;
pub mod statistics;
pub mod stream;
pub mod transform;
pub mod transforms;
pub mod utils;

// Re-exports
pub use builder::DatasetBuilder;
pub use config::DatasetConfig;
pub use core::{Batch, Item};
pub use dataset::Dataset;

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_dataset::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_dataset::version_string;
/// assert_eq!(version_string(), "brain-dataset v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-dataset v{}", VERSION)
}

/// Standard prelude imports for dataset pipelines.
///
/// ```rust
/// use brain_dataset::prelude::*;
/// let item = Item::new(1, Tensor::zeros(vec![1]));
/// assert_eq!(item.id, 1);
/// ```
pub mod prelude {
    pub use crate::builder::DatasetBuilder;
    pub use crate::config::DatasetConfig;
    pub use crate::core::{Batch, Item};
    pub use crate::dataset::Dataset;
    pub use brain_core::Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
