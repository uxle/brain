//! # Quantized ONNX Support (QuantizeLinear / DequantizeLinear)
//!
//! Int8 affine quantization, Q/DQ node conversion, and scale/zero-point unpacking.
#![allow(missing_docs)]

use crate::ir::OnnxModel;

/// Configuration for ONNX INT8 quantization.
#[derive(Debug, Clone, Default)]
pub struct QuantizeOnnxConfig {
    pub per_channel: bool,
    pub symmetric: bool,
}

/// Inspects an OnnxModel to check for QuantizeLinear / DequantizeLinear operators.
pub fn has_quantized_nodes(model: &OnnxModel) -> bool {
    model.graph.nodes.iter().any(|n| n.op_type == "QuantizeLinear" || n.op_type == "DequantizeLinear")
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
