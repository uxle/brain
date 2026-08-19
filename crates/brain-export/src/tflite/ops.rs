//! # TFLite Builtin Op Codes & Options
//!
//! Numeric op codes for standard TensorFlow Lite runtime kernels.

/// Maps standard op identifier to TFLite builtin operator opcode integer.
pub fn map_to_tflite_builtin_code(op_name: &str) -> Option<i32> {
    match op_name {
        "Add" => Some(0),
        "Conv2d" => Some(3),
        "DepthwiseConv2d" => Some(4),
        "Mul" => Some(18),
        "Relu" => Some(19),
        "Reshape" => Some(22),
        "Softmax" => Some(25),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
