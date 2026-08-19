//! # Brain Export — Universal Multi-Format Model Deployment Engine
//!
//! Provides production-grade export pipelines to ONNX, TFLite, CoreML, and WebNN formats.
//!
//! ## Subsystems
//!
//! * [`model`] - Universal `ExportModel` abstraction and exporter traits
//! * [`onnx`] - Standalone binary ONNX protobuf generator and graph checker
//! * [`tflite`] - FlatBuffers binary TFLite generator and operator kernels
//! * [`coreml`] - Apple CoreML (.mlpackage) binary generator
//! * [`webnn`] - W3C WebNN JSON graph descriptor generator
//! * [`common`] - Intermediate Export IR, weight serializers, and dtype mapping
//! * [`convert`] - Format-to-format conversion routines
//! * [`verify`] - Numerical round-trip output verification
//! * [`export_all`] - Bulk export orchestrator with manifest generation
//! * [`zip`] - Pure Rust minimal ZIP packaging engine
//!
//! ## Quick Start Example
//!
//! ```rust
//! use brain_export::prelude::*;
//!
//! let opt = ExportOptions::default();
//! assert_eq!(opt.format, ExportFormat::Onnx);
//! ```

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::identity_op,
    clippy::derivable_impls,
    clippy::manual_clamp,
    clippy::type_complexity,
    clippy::manual_is_multiple_of
)]

pub mod builder;
pub mod common;
pub mod config;
pub mod convert;
pub mod core;
pub mod coreml;
pub mod export_all;
pub mod r#impl;
pub mod model;
pub mod name_gen;
pub mod onnx;
pub mod ops;
pub mod ops_supported;
pub mod quant_export;
pub mod tflite;
pub mod utils;
pub mod verify;
pub mod webnn;
pub mod zip;

// Re-exports
pub use builder::ExportBuilder;
pub use config::{ExportConfig, TargetPlatform};
pub use core::{ExportError, ExportFormat, ExportOptions};
pub use model::{ExportModel, ModelExporter};

/// Package version string.
pub const VERSION: &str = "0.2.0";
pub const MAJOR_VERSION: u32 = 0;
pub const MINOR_VERSION: u32 = 2;
pub const PATCH_VERSION: u32 = 0;

/// Returns the crate version triple.
///
/// ```rust
/// use brain_export::version_tuple;
/// assert_eq!(version_tuple(), (0, 2, 0));
/// ```
pub fn version_tuple() -> (u32, u32, u32) {
    (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)
}

/// Returns a formatted version string.
///
/// ```rust
/// use brain_export::version_string;
/// assert_eq!(version_string(), "brain-export v0.2.0");
/// ```
pub fn version_string() -> String {
    format!("brain-export v{}", VERSION)
}

/// Standard prelude imports for model export.
///
/// ```rust
/// use brain_export::prelude::*;
/// let opt = ExportOptions::default();
/// assert_eq!(opt.format, ExportFormat::Onnx);
/// ```
pub mod prelude {
    pub use crate::builder::ExportBuilder;
    pub use crate::config::{ExportConfig, TargetPlatform};
    pub use crate::core::{ExportError, ExportFormat, ExportOptions};
    pub use crate::model::{ExportModel, ModelExporter};
    pub use brain_core::Tensor;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
