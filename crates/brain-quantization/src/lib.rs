//! # Brain Quantization & Sparsity Framework (`brain-quantization`)
//!
//! Production-grade post-training dynamic/static quantization, Quantization-Aware Training (QAT),
//! magnitude and structured pruning, sparse matrix linear algebra (CSR/CSC/COO), and mixed precision.
#![allow(missing_docs)]

pub mod core;
pub mod config;
pub mod utils;
pub mod dtype_map;
pub mod calibration;
pub mod quantizer;
pub mod quant_dynamic;
pub mod quant_static;
pub mod fake_quant;
pub mod qlinear;
pub mod qconv;
pub mod qmatmul;
pub mod ops;
pub mod prune;
pub mod sparse;
pub mod block_quant;
pub mod act_quant;
pub mod mixed;
pub mod graph_quant;
pub mod error_analysis;
pub mod bench_quant;
pub mod runtime;
pub mod builder;
pub mod helper;
pub mod r#impl;

pub use core::{QParams, QuantDType, QuantError, QuantResult, QuantScheme, QuantTensor};
pub use config::{BlockQuantConfig, DynamicConfig, FakeQuantConfig, PruneConfig, QuantConfig, SparseConfig, StaticConfig};
pub use calibration::{CalibrationConfig, CalibrationMethod, EntropyObserver, MinMaxObserver, MovingAverageObserver, Observer, PercentileObserver};
pub use quantizer::{AffineQuantizer, Quantizer, QuantizerKind, SymmetricQuantizer};
pub use quant_dynamic::DynamicQuantizer;
pub use quant_static::StaticQuantizer;
pub use fake_quant::FakeQuantize;
pub use qlinear::{QLinear, QLinearConfig};
pub use qconv::{QConv2d, QConvConfig};
pub use qmatmul::{q8_matmul, QMatMulConfig};
pub use prune::{MagnitudePruner, Pruner, PruneResult, StructuredPruner, schedule::IterativePruneSchedule};
pub use sparse::{CsrMatrix, ops::{spmm, spmv}};
pub use block_quant::BlockQuantizer;
pub use act_quant::{ActQuantConfig, ActQuantizer};
pub use mixed::{MixedConfig, MixedPrecisionQuantizer};
pub use graph_quant::{GraphQuantConfig, GraphQuantizer};
pub use error_analysis::{analyze_quantization_error, QuantErrorReport};
pub use bench_quant::{QuantBench, QuantBenchReport};
pub use runtime::QuantRuntime;
pub use builder::{PipelineMode, QuantBuilder};
pub use r#impl::{apply_magnitude_prune, dequantize_tensor, quantize_tensor};

/// Semantic version of the `brain-quantization` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and structs.
pub mod prelude {
    pub use super::core::{QParams, QuantDType, QuantScheme, QuantTensor};
    pub use super::config::{DynamicConfig, FakeQuantConfig, PruneConfig, QuantConfig, StaticConfig};
    pub use super::quantizer::{AffineQuantizer, Quantizer, SymmetricQuantizer};
    pub use super::quant_dynamic::DynamicQuantizer;
    pub use super::quant_static::StaticQuantizer;
    pub use super::fake_quant::FakeQuantize;
    pub use super::qlinear::QLinear;
    pub use super::qconv::QConv2d;
    pub use super::prune::{MagnitudePruner, Pruner};
    pub use super::sparse::CsrMatrix;
    pub use super::builder::QuantBuilder;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
