//! # Brain Quantization & Sparsity Framework (`brain-quantization`)
//!
//! Production-grade post-training dynamic/static quantization, Quantization-Aware Training (QAT),
//! magnitude and structured pruning, sparse matrix linear algebra (CSR/CSC/COO), and mixed precision.
#![allow(missing_docs)]

pub mod act_quant;
pub mod bench_quant;
pub mod block_quant;
pub mod builder;
pub mod calibration;
pub mod config;
pub mod core;
pub mod dtype_map;
pub mod error_analysis;
pub mod fake_quant;
pub mod graph_quant;
pub mod helper;
pub mod r#impl;
pub mod mixed;
pub mod ops;
pub mod prune;
pub mod qconv;
pub mod qlinear;
pub mod qmatmul;
pub mod quant_dynamic;
pub mod quant_static;
pub mod quantizer;
pub mod runtime;
pub mod sparse;
pub mod utils;

pub use act_quant::{ActQuantConfig, ActQuantizer};
pub use bench_quant::{QuantBench, QuantBenchReport};
pub use block_quant::BlockQuantizer;
pub use builder::{PipelineMode, QuantBuilder};
pub use calibration::{
    CalibrationConfig, CalibrationMethod, EntropyObserver, MinMaxObserver, MovingAverageObserver,
    Observer, PercentileObserver,
};
pub use config::{
    BlockQuantConfig, DynamicConfig, FakeQuantConfig, PruneConfig, QuantConfig, SparseConfig,
    StaticConfig,
};
pub use core::{QParams, QuantDType, QuantError, QuantResult, QuantScheme, QuantTensor};
pub use error_analysis::{analyze_quantization_error, QuantErrorReport};
pub use fake_quant::FakeQuantize;
pub use graph_quant::{GraphQuantConfig, GraphQuantizer};
pub use mixed::{MixedConfig, MixedPrecisionQuantizer};
pub use prune::{
    schedule::IterativePruneSchedule, MagnitudePruner, PruneResult, Pruner, StructuredPruner,
};
pub use qconv::{QConv2d, QConvConfig};
pub use qlinear::{QLinear, QLinearConfig};
pub use qmatmul::{q8_matmul, QMatMulConfig};
pub use quant_dynamic::DynamicQuantizer;
pub use quant_static::StaticQuantizer;
pub use quantizer::{AffineQuantizer, Quantizer, QuantizerKind, SymmetricQuantizer};
pub use r#impl::{apply_magnitude_prune, dequantize_tensor, quantize_tensor};
pub use runtime::QuantRuntime;
pub use sparse::{
    ops::{spmm, spmv},
    CsrMatrix,
};

/// Semantic version of the `brain-quantization` crate.
pub const VERSION: &str = "0.2.0";

/// Convenient prelude re-exporting key traits and structs.
pub mod prelude {
    pub use super::builder::QuantBuilder;
    pub use super::config::{
        DynamicConfig, FakeQuantConfig, PruneConfig, QuantConfig, StaticConfig,
    };
    pub use super::core::{QParams, QuantDType, QuantScheme, QuantTensor};
    pub use super::fake_quant::FakeQuantize;
    pub use super::prune::{MagnitudePruner, Pruner};
    pub use super::qconv::QConv2d;
    pub use super::qlinear::QLinear;
    pub use super::quant_dynamic::DynamicQuantizer;
    pub use super::quant_static::StaticQuantizer;
    pub use super::quantizer::{AffineQuantizer, Quantizer, SymmetricQuantizer};
    pub use super::sparse::CsrMatrix;
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
