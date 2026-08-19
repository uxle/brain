//! # ONNX Intermediate Representation (IR)
//!
//! Sanitized graph representation: `OnnxModel`, `OnnxGraph`, `OnnxNode`, `OnnxValue`.
#![allow(missing_docs)]

use std::collections::HashMap;
use brain_core::Tensor;

/// Canonical ONNX Node in intermediate representation.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, String>,
}

/// Canonical ONNX Value (tensor variable / tensor edge).
#[derive(Debug, Clone)]
pub struct OnnxValue {
    pub name: String,
    pub shape: Vec<usize>,
    pub is_initializer: bool,
    pub tensor_data: Option<Tensor>,
}

/// Canonical ONNX Graph IR.
#[derive(Debug, Clone, Default)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub values: HashMap<String, OnnxValue>,
}

/// Canonical Top-Level ONNX Model IR.
#[derive(Debug, Clone, Default)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub graph: OnnxGraph,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
