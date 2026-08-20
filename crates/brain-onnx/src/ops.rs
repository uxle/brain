//! # ONNX Operator Registry & Opset Compatibility
//!
//! Standard ONNX operator taxonomy, domain definitions, and opset availability matrix.
#![allow(missing_docs)]

/// Standard ONNX operator metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpSpec {
    pub name: &'static str,
    pub domain: &'static str,
    pub min_opset: i64,
    pub max_opset: i64,
}

pub const STANDARD_OPS: &[OpSpec] = &[
    OpSpec {
        name: "Add",
        domain: "ai.onnx",
        min_opset: 7,
        max_opset: 21,
    },
    OpSpec {
        name: "Sub",
        domain: "ai.onnx",
        min_opset: 7,
        max_opset: 21,
    },
    OpSpec {
        name: "Mul",
        domain: "ai.onnx",
        min_opset: 7,
        max_opset: 21,
    },
    OpSpec {
        name: "Div",
        domain: "ai.onnx",
        min_opset: 7,
        max_opset: 21,
    },
    OpSpec {
        name: "MatMul",
        domain: "ai.onnx",
        min_opset: 9,
        max_opset: 21,
    },
    OpSpec {
        name: "Gemm",
        domain: "ai.onnx",
        min_opset: 9,
        max_opset: 21,
    },
    OpSpec {
        name: "Conv",
        domain: "ai.onnx",
        min_opset: 1,
        max_opset: 21,
    },
    OpSpec {
        name: "Relu",
        domain: "ai.onnx",
        min_opset: 6,
        max_opset: 21,
    },
    OpSpec {
        name: "Sigmoid",
        domain: "ai.onnx",
        min_opset: 6,
        max_opset: 21,
    },
    OpSpec {
        name: "Tanh",
        domain: "ai.onnx",
        min_opset: 6,
        max_opset: 21,
    },
    OpSpec {
        name: "Softmax",
        domain: "ai.onnx",
        min_opset: 1,
        max_opset: 21,
    },
    OpSpec {
        name: "Reshape",
        domain: "ai.onnx",
        min_opset: 5,
        max_opset: 21,
    },
    OpSpec {
        name: "Transpose",
        domain: "ai.onnx",
        min_opset: 1,
        max_opset: 21,
    },
    OpSpec {
        name: "Concat",
        domain: "ai.onnx",
        min_opset: 4,
        max_opset: 21,
    },
    OpSpec {
        name: "BatchNormalization",
        domain: "ai.onnx",
        min_opset: 9,
        max_opset: 21,
    },
    OpSpec {
        name: "GlobalAveragePool",
        domain: "ai.onnx",
        min_opset: 1,
        max_opset: 21,
    },
    OpSpec {
        name: "Flatten",
        domain: "ai.onnx",
        min_opset: 9,
        max_opset: 21,
    },
    OpSpec {
        name: "Constant",
        domain: "ai.onnx",
        min_opset: 1,
        max_opset: 21,
    },
];

/// Checks whether an operator is supported in a given opset version.
pub fn is_op_supported(op_name: &str, opset_version: i64) -> bool {
    for spec in STANDARD_OPS {
        if spec.name == op_name {
            return opset_version >= spec.min_opset && opset_version <= spec.max_opset;
        }
    }
    false
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
