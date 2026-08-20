//! # Brain Core
//!
//! Core primitives, data structures, device representations, and tensor computation engine
//! for the Brain deep learning framework.
//!
//! ## Architecture & Modules
//!
//! - [`device`]: Device abstractions (CPU, CUDA, MPS, Vulkan) and thread-local device stack.
//! - [`dtype`]: Data type system (F64, F32, F16, BF16, I64, I32, I16, I8, U8, Bool) and promotion rules.
//! - [`error`]: Robust error handling, error categories, and chained context.
//! - [`memory`]: Aligned memory buffers, memory arenas, buddy allocators, and binned pools.
//! - [`random`]: Deterministic PRNGs (XORShift128+, PCG32, SplitMix64, ChaCha8) and distributions.
//! - [`serialization`]: Binary checkpoint format v2 with CRC32 integrity and multi-tensor archives.
//! - [`shape`]: Multi-dimensional shapes, dimension algebra, and broadcast resolution.
//! - [`tensor`]: N-dimensional tensor implementation, BLAS routines, autograd, and neural operators.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::all,
    unused_variables,
    unused_mut,
    dead_code,
    unused_imports,
    deprecated
)]

pub mod brain_mind;
pub mod device;
pub mod dtype;
pub mod error;
pub mod memory;
pub mod pool;
pub mod random;
pub mod serialization;
pub mod shape;
pub mod tensor;

// =============================================================================
// Re-exports
// =============================================================================

pub use brain_mind::{BrainMind, TeachSummary};
pub use device::{Backend, CpuBackend, Device, DeviceInfo, DeviceType, SimdCpuBackend};
pub use dtype::{DType, DTypeInfo};
pub use error::{BrainError, BrainResult};
pub use serialization::{BrainModelFile, NodeCoord3D, TensorArchive, BN_MAGIC};
pub use shape::{Dim, Shape, Strides};
pub use tensor::{Tensor, TensorStats};

// =============================================================================
// Prelude
// =============================================================================

/// Convenience re-exports of common traits and types.
pub mod prelude {
    pub use crate::brain_mind::{BrainMind, TeachSummary};
    pub use crate::device::{Backend, CpuBackend, Device, DeviceType, SimdCpuBackend};
    pub use crate::dtype::DType;
    pub use crate::error::{BrainError, BrainResult};
    pub use crate::random::{self, BrainRng, Rng};
    pub use crate::serialization::{BrainModelFile, NodeCoord3D, TensorArchive, BN_MAGIC};
    pub use crate::shape::{Dim, Shape, Strides};
    pub use crate::tensor::{Tensor, TensorStats};
}

// =============================================================================
// Framework Configuration & Diagnostics
// =============================================================================

/// Global framework configuration settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Default device placement.
    pub default_device: Device,
    /// Default floating point precision.
    pub default_dtype: DType,
    /// Number of worker threads for parallel BLAS execution.
    pub num_threads: usize,
    /// Whether deterministic PRNG execution is enforced.
    pub deterministic: bool,
    /// Memory pool alignment in bytes.
    pub default_alignment: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            default_device: Device::Cpu,
            default_dtype: DType::F64,
            num_threads: 1,
            deterministic: false,
            default_alignment: 64,
        }
    }
}

/// Returns the framework version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns framework build information.
pub fn build_info() -> &'static str {
    concat!(
        "Brain Core v",
        env!("CARGO_PKG_VERSION"),
        " (Rust Edition 2021, std-only)"
    )
}

/// Formats a byte count into human-readable string (B, KB, MB, GB).
pub fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let b = bytes as f64;
    if b < KB {
        format!("{} B", bytes)
    } else if b < MB {
        format!("{:.2} KB", b / KB)
    } else if b < GB {
        format!("{:.2} MB", b / MB)
    } else {
        format!("{:.2} GB", b / GB)
    }
}

/// Formats a shape slice into a human-readable string: `[2, 3, 4]`.
pub fn format_shape(dims: &[usize]) -> String {
    let parts: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_metadata() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
        assert!(build_info().contains("Brain Core"));
    }

    #[test]
    fn test_format_bytes_table() {
        let cases = [
            (0, "0 B"),
            (500, "500 B"),
            (1024, "1.00 KB"),
            (1536, "1.50 KB"),
            (1024 * 1024, "1.00 MB"),
            (1024 * 1024 * 1024, "1.00 GB"),
            (5 * 1024 * 1024 * 1024, "5.00 GB"),
        ];
        for (bytes, expected) in cases {
            assert_eq!(format_bytes(bytes), expected);
        }
    }

    #[test]
    fn test_format_shape_table() {
        let cases: &[(&[usize], &str)] = &[
            (&[], "[]"),
            (&[5], "[5]"),
            (&[2, 3], "[2, 3]"),
            (&[1, 3, 224, 224], "[1, 3, 224, 224]"),
        ];
        for (shape, expected) in cases {
            assert_eq!(format_shape(shape), *expected);
        }
    }

    #[test]
    fn test_config_default_and_custom() {
        let def = Config::default();
        assert_eq!(def.default_device, Device::Cpu);
        assert_eq!(def.default_dtype, DType::F64);
        assert_eq!(def.num_threads, 1);
        assert!(!def.deterministic);
        assert_eq!(def.default_alignment, 64);

        let custom = Config {
            default_device: Device::cuda(0),
            default_dtype: DType::F32,
            num_threads: 8,
            deterministic: true,
            default_alignment: 128,
        };
        assert_ne!(def, custom);
    }
}
