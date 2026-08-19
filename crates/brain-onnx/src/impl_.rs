//! # High-Level ONNX Entrypoints
//!
//! `load_onnx`, `load_onnx_bytes`, `save_onnx`, and `import_and_optimize` workflow helpers.
#![allow(missing_docs)]

use super::core::OnnxResult;
use super::config::{ImportConfig, OptimizeConfig};
use super::ir::OnnxModel;
use super::import::import_model;
use super::optimize::optimize_model;
use brain_graph::GraphIr;

/// Imports and compiles an ONNX model from raw binary bytes into a optimized Brain Graph IR.
pub fn import_and_optimize(
    bytes: &[u8],
    import_cfg: &ImportConfig,
    opt_cfg: &OptimizeConfig,
) -> OnnxResult<(OnnxModel, GraphIr)> {
    let model = import_model(bytes, import_cfg)?;
    let opt_model = optimize_model(&model, opt_cfg)?;
    let graph_ir = super::ir2graph::lower_to_graph_ir(&opt_model)?;
    Ok((opt_model, graph_ir))
}

/// Loads and compiles an ONNX model from a file path.
pub fn load_onnx(path: &str) -> OnnxResult<(OnnxModel, GraphIr)> {
    let bytes = std::fs::read(path).map_err(|e| super::core::OnnxError::IoError(e.to_string()))?;
    import_and_optimize(&bytes, &ImportConfig::default(), &OptimizeConfig::default())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
