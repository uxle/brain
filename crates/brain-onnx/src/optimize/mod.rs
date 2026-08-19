//! # ONNX Graph Optimization Pipeline
//!
//! Optimization passes: constant folding, operator fusion, dead node removal, and layout normalization.
#![allow(missing_docs)]

pub mod onnx_passes;
pub use onnx_passes::{fuse_conv_relu, fuse_matmul_add, fold_constant_nodes};

use crate::core::OnnxResult;
use crate::config::OptimizeConfig;
use crate::ir::OnnxModel;

/// Optimizes an OnnxModel in-place based on OptimizeConfig.
pub fn optimize_model(model: &OnnxModel, config: &OptimizeConfig) -> OnnxResult<OnnxModel> {
    let mut optimized = model.clone();

    if config.fuse_conv_relu {
        fuse_conv_relu(&mut optimized);
    }
    if config.fuse_gemm {
        fuse_matmul_add(&mut optimized);
    }

    fold_constant_nodes(&mut optimized);

    Ok(optimized)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
