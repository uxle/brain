//! # ONNX Operator Schema Mapping
//!
//! Maps internal arithmetic, linear algebra, convolution, and activation ops to standard ONNX schemas.

/// Maps standard op identifier to ONNX op type string.
pub fn map_to_onnx_op(op_name: &str) -> Option<&'static str> {
    match op_name {
        "Add" => Some("Add"),
        "Sub" => Some("Sub"),
        "Mul" => Some("Mul"),
        "Div" => Some("Div"),
        "MatMul" => Some("MatMul"),
        "Conv2d" => Some("Conv"),
        "Relu" => Some("Relu"),
        "Softmax" => Some("Softmax"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
